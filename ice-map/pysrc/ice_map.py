import ee
import os
import toml
import folium
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def config_parser():
    # 获取脚本所在目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的完整路径
    config_path = os.path.join(os.path.dirname(script_dir), "config", "config.toml")
    with open(config_path, "r") as f:
        config = toml.load(f)
    region_file = config["region_config"]
    # 如果region_file是相对路径，也需要转换为绝对路径
    if not os.path.isabs(region_file):
        region_file = os.path.join(os.path.dirname(script_dir), "config", region_file)
    with open(region_file, "r") as f:
        region_config = toml.load(f)
    # 合并配置文件
    config.update(region_config)
    return config

def earth_engine_init(config: dict):
    ee.Initialize(project=config["project_name"])

def get_snow_data() -> tuple[dict, dict]:
    """
    处理配置信息，获取雪盖数据
    返回值一个元组, 元组中第一个元素是雪盖数据, 第二个元素是配置信息
        snow_data(dict): 雪盖数据
        config(dict): 配置信息
    """

    config = config_parser()
    earth_engine_init(config)

    dataset_name = config["dataset_name"]
    years = config["years"]
    start_date_template = config["start_date"]
    end_date_template = config["end_date"]
    NDSI_threshold = config["NDSI_threshold"]
    center_point = config["center"]
    radius = config["radius"]
    resolution = config["resolution"]
    max_pixels = config["max_pixels"]

    snow_data = {}
    roi = ee.Geometry.Point(center_point).buffer(radius)
    
    for year in years:
        year_start_date = f"{year}-{start_date_template}"
        year_end_date = f"{year}-{end_date_template}"

        # 获取MODIS雪盖数据
        modis_snow = ee.ImageCollection(dataset_name).filter(ee.Filter.date(year_start_date, year_end_date))
        # 计算年度平均雪盖
        # TODO 这里实际可以采用多种统计方法(max, min etc.)
        annual_snow = modis_snow.select('NDSI_Snow_Cover').mean()
        # 计算雪盖面积占比
        snow = annual_snow.gt(NDSI_threshold)  # NDSI>0为雪盖

        # 计算雪盖面积
        area = snow.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=resolution,
            maxPixels=max_pixels,
            bestEffort=True
        )
        
        # 计算总面积
        total_area = ee.Image.pixelArea().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=resolution,
            maxPixels=max_pixels,
            bestEffort=True
        )

        snow_data[year] = {
            'snow_area': area.get('NDSI_Snow_Cover'),
            'total_area': total_area.get('area'),
            'image': annual_snow                     
        }
    
    return (snow_data, config)

def create_map_html(snow_data: dict, config: dict):
    # 定义可视化参数
    # TODO 可视化参数写入配置文件
    vis_params = {
        'min': 0,
        'max': 100,
        'palette': ['black', 'blue', 'cyan', 'white']
    }

    # 创建底图
    center_point = config["center"]
    ice_map = folium.Map(
        location=center_point, 
        zoom_start=10,
        tiles=config["map_tiles"]
    )

    # 为每个年份添加图层
    years = config["years"]
    for year in years:
        snow_image_by_year = snow_data[year]['image']
        # 添加到地图
        map_id = snow_image_by_year.getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr=f'Google Earth Engine - MODIS Snow Cover {year}',
            overlay=True,
            name=f'年份 {year}'
        ).add_to(ice_map)

    # 添加圆形区域边界
    folium.Circle(
        location=[center_point[1], center_point[0]],  # folium需要[纬度,经度]格式
        radius=config["radius"],  # 半径（米）
        color='red',
        fill=True,
        fill_opacity=0.2,
        weight=2,
        popup=f'研究区域 (半径: {config["radius"]/1000}公里)'
    ).add_to(ice_map)

    # 添加图层控制
    folium.LayerControl().add_to(ice_map)
    
    # 保存地图
    map_file_name = config["map_file_name"]
    map_file_path = config["map_file_path"]
    
    # 确保输出目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_map_path = os.path.join(os.path.dirname(script_dir), map_file_path.strip('./'))
    os.makedirs(absolute_map_path, exist_ok=True)
    
    output_path = os.path.join(absolute_map_path, map_file_name)
    ice_map.save(output_path)
    logging.info(f"地图已保存至 {output_path}")


if __name__ == "__main__":
    snow_data, config = get_snow_data()
    # 配置日志
    logging.basicConfig(level=config["log_level"])
    create_map_html(snow_data, config)

