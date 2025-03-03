import ee
import os
import toml
import folium
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC']  # Mac系统常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建logger
logger = logging.getLogger('ice_map')

def setup_logging(config: dict):
    """
    设置日志配置
    """
    # 创建handler
    console_handler = logging.StreamHandler()
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # 添加handler
    logger.addHandler(console_handler)
    # 设置日志级别
    logger.setLevel(config.get("log_level", "INFO"))
    # 避免日志重复
    logger.propagate = False

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

def get_snow_data(config: dict) -> dict:
    """
    处理配置信息，获取雪盖数据
    返回值一个元组, 元组中第一个元素是雪盖数据, 第二个元素是配置信息
        snow_data(dict): 雪盖数据
        config(dict): 配置信息
    """

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
    
    return snow_data

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
    logger.info(f"地图已保存至 {output_path}")

# 绘制冰川覆盖率随着时间的变化图
# 柱状图+折线图
def plot_ice_coverage_chart(snow_data: dict, config: dict):
    logger.info("准备绘制冰川覆盖率随着时间的变化图...")
    
    years = config["years"]
    coverage_data = []
    for year in years:
        snow_area = ee.Number(snow_data[year]['snow_area']).getInfo()
        total_area = ee.Number(snow_data[year]['total_area']).getInfo()
        percentage = (snow_area / total_area) * 100
        coverage_data.append(percentage)
        logger.info(f"{year}年: 雪盖面积 = {snow_area/1e6:.2f} 平方公里, 占比 = {percentage:.2f}%")
    
    # 设置图表样式
    plt.style.use('bmh')  # 使用内置的bmh样式，这是一个美观的样式
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 设置背景色
    ax.set_facecolor('#f0f0f0')
    fig.patch.set_facecolor('white')
    
    # 绘制柱状图和折线图
    bar_width = 0.6
    bars = ax.bar(years, coverage_data, bar_width, 
                 alpha=0.7, 
                 color='#5B9BD5',  # 使用微软PPT风格的蓝色
                 label='年度覆盖率',
                 edgecolor='white',  # 白色边框
                 linewidth=1)
    
    line = ax.plot(years, coverage_data, 
                  color='#ED7D31',  # 使用微软PPT风格的橙色
                  linewidth=2.5, 
                  marker='o', 
                  markersize=10,
                  markerfacecolor='white',  # 点的填充色为白色
                  markeredgecolor='#ED7D31',  # 点的边框色与线条同色
                  markeredgewidth=2,
                  label='变化趋势', 
                  zorder=5)
    
    # 设置标题和标签
    ax.set_title('南极冰雪覆盖率年际变化', fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('年份', fontsize=12, labelpad=10)
    ax.set_ylabel('覆盖率 (%)', fontsize=12, labelpad=10)
    
    # 设置x轴刻度为整数年份
    ax.set_xticks(years)
    ax.set_xticklabels([str(int(year)) for year in years], fontsize=11)
    
    # 自动调整y轴范围，使差距更明显
    min_value = min(coverage_data)
    max_value = max(coverage_data)
    y_margin = (max_value - min_value) * 0.15  # 留出15%的边距
    ax.set_ylim(min_value - y_margin, max_value + y_margin)
    
    # 设置y轴刻度字体
    ax.tick_params(axis='y', labelsize=11)
    
    # 添加数据标签
    for bar, value in zip(bars, coverage_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}%',
                ha='center', va='bottom',
                fontsize=11,
                fontweight='bold',
                color='#333333')
    
    # 添加图例
    ax.legend(loc='upper right', 
             fontsize=11, 
             frameon=True,
             facecolor='white',
             edgecolor='none',
             shadow=True)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片到pic目录
    pic_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pic")
    os.makedirs(pic_dir, exist_ok=True)
    pic_path = os.path.join(pic_dir, f"{config['pic_name']}.png")
    plt.savefig(pic_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"图片已保存至 {pic_path}")
    
    # 显示图表
    plt.show()
    
if __name__ == "__main__":
    config = config_parser()
    setup_logging(config)
    snow_data = get_snow_data(config)
    create_map_html(snow_data, config)
    plot_ice_coverage_chart(snow_data, config)

