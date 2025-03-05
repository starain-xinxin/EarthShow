import ee
import os
import toml
import folium
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
import numpy as np
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC']  # Mac系统常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建logger
logger = logging.getLogger('ch4_map')

def setup_logging(config: dict):
    """
    设置日志配置，同时输出到控制台和文件
    """
    # 创建日志目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(os.path.dirname(script_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{config['experiment_id']}_{timestamp}.log")
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台显示INFO及以上级别
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG及以上级别
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 清除已存在的处理器
    logger.handlers.clear()
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 设置日志级别
    logger.setLevel(logging.DEBUG)  # 设置为最低级别，让处理器决定显示级别
    
    # 避免日志重复
    logger.propagate = False
    
    logger.info(f"日志文件已创建: {log_file}")

def config_parser():
    # 获取脚本所在目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的完整路径
    config_path = os.path.join(os.path.dirname(script_dir), "config", "config.toml")
    with open(config_path, "r") as f:
        config = toml.load(f)
    
    # 根据region_type选择正确的配置文件
    if config["region_type"] == "global":
        region_file = "region_global.toml"
    else:
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
    logger.info(f"Google Earth Engine init success, project name: {config['project_name']}")

def get_ch4_data(config: dict) -> dict:
    """
    处理配置信息，获取CH4数据
    返回值是一个字典，包含每年的CH4数据
        ch4_data(dict): CH4数据
    """
    earth_engine_init(config)

    dataset_name = config["dataset_name"]
    years = config["years"]
    start_date_template = config["start_date"]
    end_date_template = config["end_date"]
    ch4_threshold = config["ch4_threshold"]
    resolution = config["resolution"]
    max_pixels = config["max_pixels"]

    ch4_data = {}
    
    # 根据region_type选择不同的处理逻辑
    if config["region_type"] == "global":
        bbox = config["bbox"]
        grid_size = config.get("grid_size", 10)
        sample_points = config.get("sample_points", 10)
        sample_region_size = config.get("sample_region_size", 20)
        roi = ee.Geometry.Rectangle(bbox)
        is_global = True
        
        # 生成随机采样点
        random.seed(42)  # 设置随机种子以保证结果可重复
        sample_regions = []
        
        # 确保采样区域不会超出边界
        max_lat = 90 - sample_region_size
        max_lon = 180 - sample_region_size
        
        for _ in range(sample_points):
            lat = random.uniform(-max_lat, max_lat)
            lon = random.uniform(-max_lon, max_lon)
            region = {
                'bbox': [lon, lat, lon + sample_region_size, lat + sample_region_size],
                'center': [lon + sample_region_size/2, lat + sample_region_size/2]
            }
            sample_regions.append(region)
            
        logger.info(f"已生成 {sample_points} 个随机采样区域")
    else:
        center_point = config["center"]
        radius = config["radius"]
        roi = ee.Geometry.Point(center_point).buffer(radius)
        is_global = False
    
    for year in years:
        year_start_date = f"{year}-{start_date_template}"
        year_end_date = f"{year}-{end_date_template}"

        # 获取Sentinel-5P CH4数据
        ch4_collection = ee.ImageCollection(dataset_name) \
            .filter(ee.Filter.date(year_start_date, year_end_date)) \
            .select('CH4_column_volume_mixing_ratio_dry_air')
        
        # 计算年度平均CH4浓度
        annual_ch4 = ch4_collection.mean()
        
        if is_global:
            # 全球模式：使用采样统计
            grid_stats = []
            valid_values = []
            
            # 对每个采样区域进行统计
            for region in sample_regions:
                grid_bbox = ee.Geometry.Rectangle(region['bbox'])
                grid_stat = annual_ch4.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=grid_bbox,
                    scale=resolution,
                    maxPixels=max_pixels,
                    bestEffort=True
                )
                ch4_value = grid_stat.get('CH4_column_volume_mixing_ratio_dry_air').getInfo()
                grid_stats.append({
                    'lat': region['center'][1],
                    'lon': region['center'][0],
                    'ch4': ch4_value
                })
                # 收集有效值用于计算全球平均
                if ch4_value is not None:
                    valid_values.append(ch4_value)
                    logger.info(f"{year}年 采样点 [{region['center'][0]:.1f}, {region['center'][1]:.1f}] CH4浓度: {ch4_value:.1f} ppb")
            
            # 计算全球平均值（使用所有有效采样点的平均）
            mean_ch4 = sum(valid_values) / len(valid_values) if valid_values else None
            
            ch4_data[year] = {
                'mean_ch4': mean_ch4,
                'image': annual_ch4,
                'grid_stats': grid_stats
            }
            
        else:
            # 局部区域模式：直接计算区域平均
            stats = annual_ch4.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=resolution,
                maxPixels=max_pixels,
                bestEffort=True
            )
            
            ch4_data[year] = {
                'mean_ch4': stats.get('CH4_column_volume_mixing_ratio_dry_air'),
                'image': annual_ch4
            }
    
    return ch4_data

def create_map_html(ch4_data: dict, config: dict):
    # 定义可视化参数
    vis_params = {
        'min': 1750,  # ppb
        'max': 1900,  # ppb
        'palette': ['blue', 'cyan', 'yellow', 'red']  # 从低到高的颜色渐变
    }

    # 创建底图
    if config["region_type"] == "global":
        # 全球视图
        ch4_map = folium.Map(
            location=[0, 0],
            zoom_start=2,
            tiles=config["map_tiles"]
        )
    else:
        # 局部区域视图
        center_point = config["center"]
        ch4_map = folium.Map(
            location=[center_point[1], center_point[0]],
            zoom_start=8,
            tiles=config["map_tiles"]
        )

    # 为每个年份添加图层
    years = config["years"]
    for year in years:
        ch4_image_by_year = ch4_data[year]['image']
        # 添加到地图
        map_id = ch4_image_by_year.getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr=f'Google Earth Engine - Sentinel-5P CH4 {year}',
            overlay=True,
            name=f'年份 {year}'
        ).add_to(ch4_map)

    # 如果是局部区域，添加圆形边界
    if config["region_type"] != "global":
        center_point = config["center"]
        folium.Circle(
            location=[center_point[1], center_point[0]],
            radius=config["radius"],
            color='red',
            fill=True,
            fill_opacity=0.2,
            weight=2,
            popup=f'研究区域 (半径: {config["radius"]/1000}公里)'
        ).add_to(ch4_map)

    # 添加图层控制
    folium.LayerControl().add_to(ch4_map)
    
    # 保存地图
    map_file_name = config["map_file_name"]
    map_file_path = config["map_file_path"]
    
    # 确保输出目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_map_path = os.path.join(os.path.dirname(script_dir), map_file_path.strip('./'))
    os.makedirs(absolute_map_path, exist_ok=True)
    
    output_path = os.path.join(absolute_map_path, f"{config['experiment_id']}_{map_file_name}")
    ch4_map.save(output_path)
    logger.info(f"地图已保存至 {output_path}")

def plot_ch4_trend_chart(ch4_data: dict, config: dict):
    logger.info("准备绘制CH4浓度年际变化图...")
    
    years = config["years"]
    ch4_values = []
    for year in years:
        ch4_value = ee.Number(ch4_data[year]['mean_ch4']).getInfo()
        if ch4_value is None:
            logger.warning(f"{year}年: 未获取到有效的CH4数据")
            ch4_value = 0  # 或者可以选择跳过这个年份
        else:
            logger.info(f"{year}年: CH4平均浓度 = {ch4_value:.1f} ppb")
        ch4_values.append(ch4_value)
    
    # 过滤掉无效的数据点
    valid_years = []
    valid_values = []
    for year, value in zip(years, ch4_values):
        if value > 0:  # 只保留有效值
            valid_years.append(year)
            valid_values.append(value)
    
    if not valid_values:
        logger.error("没有获取到任何有效的CH4数据，无法绘制图表")
        return
    
    # 创建图表
    if config["region_type"] == "global":
        # 全球模式：创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1.2])
    else:
        # 局部区域模式：只创建一个图
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f0f0f0')
    
    # 绘制柱状图和折线图
    bar_width = 0.6
    bars = ax1.bar(valid_years, valid_values, bar_width, 
                  alpha=0.7, 
                  color='#5B9BD5',
                  label='年度平均浓度',
                  edgecolor='white',
                  linewidth=1)
    
    ax1.plot(valid_years, valid_values, 
            color='#ED7D31',
            linewidth=2.5, 
            marker='o', 
            markersize=10,
            markerfacecolor='white',
            markeredgecolor='#ED7D31',
            markeredgewidth=2,
            label='变化趋势', 
            zorder=5)
    
    # 设置标题和标签
    title_prefix = "全球" if config["region_type"] == "global" else "区域"
    ax1.set_title(f'{title_prefix}大气CH4浓度年际变化', fontsize=16, pad=20, fontweight='bold')
    ax1.set_xlabel('年份', fontsize=12, labelpad=10)
    ax1.set_ylabel('CH4浓度 (ppb)', fontsize=12, labelpad=10)
    
    # 设置x轴刻度
    ax1.set_xticks(valid_years)
    ax1.set_xticklabels([str(int(year)) for year in valid_years], fontsize=11)
    
    # 自动调整y轴范围
    min_value = min(valid_values)
    max_value = max(valid_values)
    y_margin = (max_value - min_value) * 0.15 if min_value != max_value else max_value * 0.15
    ax1.set_ylim(min_value - y_margin, max_value + y_margin)
    
    # 添加数据标签
    for bar, value in zip(bars, valid_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}',
                ha='center', va='bottom',
                fontsize=11,
                fontweight='bold',
                color='#333333')
    
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
    
    # 如果是全球模式，添加热力图
    if config["region_type"] == "global":
        ax2.set_facecolor('#f0f0f0')
        latest_year = max(valid_years)  # 使用有效数据的最新年份
        grid_data = ch4_data[latest_year]['grid_stats']
        
        # 提取网格数据
        lats = []
        lons = []
        values = []
        for grid in grid_data:
            # 直接获取ch4值，不需要转换为ee.Number
            ch4_value = grid['ch4']
            if ch4_value is not None:  # 只添加非None的值
                lats.append(grid['lat'])
                lons.append(grid['lon'])
                values.append(ch4_value)
        
        if values:  # 只在有有效数据时绘制热力图
            # 创建网格
            grid_size = config.get("grid_size", 10)
            lon_bins = np.arange(-180, 181, grid_size)
            lat_bins = np.arange(-90, 91, grid_size)
            
            # 创建热力图数据
            heat_data, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins], weights=values)
            heat_data = heat_data.T
            
            # 绘制热力图
            im = ax2.imshow(heat_data, 
                          extent=[-180, 180, -90, 90],
                          aspect='auto',
                          cmap='RdYlBu_r',
                          interpolation='nearest')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('CH4浓度 (ppb)', fontsize=12)
            
            # 设置标题和标签
            ax2.set_title(f'{latest_year}年全球CH4浓度分布', fontsize=16, pad=20, fontweight='bold')
            ax2.set_xlabel('经度', fontsize=12, labelpad=10)
            ax2.set_ylabel('纬度', fontsize=12, labelpad=10)
            
            # 添加采样点标记
            ax2.scatter(lons, lats, c='red', s=50, alpha=0.6, label='采样点')
            ax2.legend(loc='upper right')
        else:
            logger.warning(f"{latest_year}年没有有效的网格数据，跳过热力图绘制")
            ax2.text(0.5, 0.5, '无有效数据可供显示',
                    ha='center', va='center',
                    transform=ax2.transAxes,
                    fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    pic_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pic")
    os.makedirs(pic_dir, exist_ok=True)
    pic_path = os.path.join(pic_dir, f"{config['experiment_id']}_{config['pic_name']}.png")
    plt.savefig(pic_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"图片已保存至 {pic_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    config = config_parser()
    setup_logging(config)
    ch4_data = get_ch4_data(config)
    create_map_html(ch4_data, config)
    plot_ch4_trend_chart(ch4_data, config) 