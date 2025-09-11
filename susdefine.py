import pandas as pd
import numpy as np
import random

def create_categorization_functions(df):
    """Create categorization functions based on actual data distribution."""
    
    # Dictionary to store percentile thresholds for each metric
    thresholds = {}
    
    # List of metrics to categorize
    metrics = [
        'building_density', 'avg_building_size',
        'road_density_km', 'intersection_density', 'avg_land_use_area_km2',
        'road_network_complexity', 'land_use_diversity', 'unique_land_uses',
        'amenity_density'
    ]
    
    # Calculate percentiles for each metric
    for metric in metrics:
        thresholds[metric] = [
            df[metric].quantile(0.4),
            df[metric].quantile(0.6),
            df[metric].quantile(0.8),
            df[metric].quantile(0.95)
        ]
    
    # Create categorization functions using these thresholds
    def categorize_building_density(value):
        t = thresholds['building_density']
        if value < t[0]:
            return "very low", "sparse and minimal urban development with very few buildings"
        elif value < t[1]:
            return "low", "low-density suburban area with scattered buildings"
        elif value < t[2]:
            return "medium", "medium-density mixed residential area"
        elif value < t[3]:
            return "high", "high-density urban neighborhood"
        else:
            return "very high", "extremely dense urban core with concentrated buildings"

    def categorize_building_size(value):
        t = thresholds['avg_building_size']
        if value < t[0]:
            return "very small", "predominantly tiny structures"
        elif value < t[1]:
            return "small", "mostly small buildings"
        elif value < t[2]:
            return "medium", "average-sized buildings"
        elif value < t[3]:
            return "large", "substantial buildings"
        else:
            return "very large", "predominantly large structures"

    def categorize_road_density(value):
        t = thresholds['road_density_km']
        if value < t[0]:
            return "very low", "minimal road infrastructure"
        elif value < t[1]:
            return "low", "sparse road network"
        elif value < t[2]:
            return "medium", "moderate road coverage"
        elif value < t[3]:
            return "high", "extensive road network"
        else:
            return "very high", "extremely dense road system"

    def categorize_intersection_density(value):
        t = thresholds['intersection_density']
        if value < t[0]:
            return "very low", "few intersections with simple connectivity"
        elif value < t[1]:
            return "low", "limited number of intersections"
        elif value < t[2]:
            return "medium", "moderate number of intersections"
        elif value < t[3]:
            return "high", "numerous intersections creating a well-connected grid"
        else:
            return "very high", "extremely high number of intersections forming a complex network"

    def categorize_land_use_area(value):
        t = thresholds['avg_land_use_area_km2']
        if value < t[0]:
            return "very small", "appears to have smaller land use parcels"
        elif value < t[1]:
            return "small", "shows relatively small land use areas"
        elif value < t[2]:
            return "medium", "displays medium-sized land use areas"
        elif value < t[3]:
            return "large", "indicates larger land use parcels"
        else:
            return "very large", "suggests very large land use areas"

    def categorize_network_complexity(value):
        t = thresholds['road_network_complexity']
        if value < t[0]:
            return "very low", "extremely simple road layout with minimal variation"
        elif value < t[1]:
            return "low", "straightforward road pattern with limited complexity"
        elif value < t[2]:
            return "medium", "moderately complex road network"
        elif value < t[3]:
            return "high", "intricate road pattern with significant variation"
        else:
            return "very high", "highly complex and irregular road network"

    def categorize_land_use_diversity(value):
        t = thresholds['land_use_diversity']
        if value < t[0]:
            return "very low", "homogeneous land use with minimal variety"
        elif value < t[1]:
            return "low", "limited land use mix"
        elif value < t[2]:
            return "medium", "moderate mix of different land uses"
        elif value < t[3]:
            return "high", "diverse mix of land uses creating a vibrant urban fabric"
        else:
            return "very high", "extremely diverse and mixed land uses"

    def categorize_unique_land_uses(value):
        t = thresholds['unique_land_uses']
        if value < t[2]:
            return "minimal", "almost single-purpose area"
        elif value < t[3]:
            return "several", "several different land uses"
        elif value < t[3] * 2:
            return "many", "many different land uses creating diversity"
        else:
            return "extensive", "extensive variety of land uses creating a highly mixed environment"
        
    def categorize_amenity_density(value):
        t = thresholds['amenity_density']
        # Since 80% of areas have 0 amenities, we need a different approach
        if value == 0:
            return "none", "no amenities or services available"
        elif value <= t[3] / 4:  # Up to 25% of the 80th percentile value
            return "very low", "minimal amenities with only basic services"
        elif value <= t[3] / 2:  # Up to 50% of the 80th percentile value
            return "low", "limited amenities and services"
        elif value <= t[3]:  # Up to the 80th percentile
            return "medium", "moderate number of amenities"
        elif value <= t[3] * 2:  # Up to 2x the 80th percentile
            return "high", "abundant amenities and services"
        else:  # Above 2x the 80th percentile
            return "very high", "extremely high concentration of amenities and services"
    
    # Return all categorization functions
    return {
        'building_density': categorize_building_density,
        'avg_building_size': categorize_building_size,
        'road_density_km': categorize_road_density,
        'intersection_density': categorize_intersection_density,
        'avg_land_use_area_km2': categorize_land_use_area,
        'road_network_complexity': categorize_network_complexity,
        'land_use_diversity': categorize_land_use_diversity,
        'unique_land_uses': categorize_unique_land_uses,
        'amenity_density': categorize_amenity_density
    }

def generate_enhanced_prompt(row, categorize_funcs):
    """Generate a detailed prompt with embedded numerical data and descriptive text."""
    
    # Get categories and descriptions using the data-driven categorization functions
    bd_cat, bd_desc = categorize_funcs['building_density'](row['building_density'])
    bs_cat, bs_desc = categorize_funcs['avg_building_size'](row['avg_building_size'])
    rd_cat, rd_desc = categorize_funcs['road_density_km'](row['road_density_km'])
    id_cat, id_desc = categorize_funcs['intersection_density'](row['intersection_density'])
    lua_cat, lua_desc = categorize_funcs['avg_land_use_area_km2'](row['avg_land_use_area_km2'])
    rnc_cat, rnc_desc = categorize_funcs['road_network_complexity'](row['road_network_complexity'])
    lud_cat, lud_desc = categorize_funcs['land_use_diversity'](row['land_use_diversity'])
    ulu_cat, ulu_desc = categorize_funcs['unique_land_uses'](row['unique_land_uses'])
    ad_cat, ad_desc = categorize_funcs['amenity_density'](row['amenity_density'])
    
    # DENSITY section - 包含建筑密度、道路密度、交叉口密度、设施密度
    density_section = (
        f"This area has {bd_cat} building density ({row['building_density']:.1f} buildings per square kilometer), {bd_desc}. "
        f"It features {rd_cat} road density ({row['road_density_km']:.1f} km per square kilometer) where {rd_desc}, "
        f"with {id_cat} intersection density ({row['intersection_density']:.1f} intersections per square kilometer) where {id_desc}. "
        f"The area has {ad_cat} amenity density ({row['amenity_density']:.3f} amenities per square kilometer) where {ad_desc}. "
    )
    
    # DIVERSITY section - 包含土地利用多样性、独特土地利用类型数量
    diversity_section = (
        f"The neighborhood exhibits {lud_cat} land use diversity (index: {row['land_use_diversity']:.2f}) where {lud_desc}, "
        f"containing {ulu_cat} unique land uses ({row['unique_land_uses']}) where {ulu_desc}. "
    )
    
    # DESIGN section - 包含建筑尺寸、道路网络复杂度、土地利用区域大小
    design_section = (
        f"The urban design features {bs_cat} buildings (averaging {row['avg_building_size']:.1f} square meters), {bs_desc}. "
        f"The road network shows {rnc_cat} complexity ({row['road_network_complexity']:.2f}) where {rnc_desc}. "
        f"The land use areas are {lua_cat} (averaging {row['avg_land_use_area_km2']:.3f} square kilometers) where {lua_desc}. "
    )
    
    # Combine into final prompt
    prompt = (
        f"A realistic image of an urban area in Orlando--left half as satellite image, right half as corresponding land use map. "
        f"{density_section}{diversity_section}{design_section}"
        f"The image should be viewed from above, in high resolution aerial photography style, "
        f"clearly showing the urban morphology and land use patterns described."
    )
    
    return prompt

def run_sustainable_development_test():
    # 读取测试数据集
    test_df = pd.read_csv('./metrics_datagrido150/test_set_below_0.7.csv')
    
    # 指定要测试的idx
    target_indices = [1199, 11959, 13085, 13674, 15219, 15814, 16417, 16717, 16892, 16984, 18099, 18659, 20624, 21684, 4659, 4918, 6850]
    test_samples = test_df[test_df['idx'].isin(target_indices)].copy()

    # 创建分类函数
    categorize_funcs = create_categorization_functions(test_df)

    # 创建结果存储列表
    sustainable_results = []

    # 对每个指定样本进行可持续发展改造
    for _, sample in test_samples.iterrows():
        # 添加原始样本的提示
        original_prompt = generate_enhanced_prompt(sample, categorize_funcs)
        
        result_row = {
            'idx': sample['idx'],
            'test_type': 'original',
            'prompt': original_prompt
        }
        sustainable_results.append(result_row)
        
        # 创建可持续发展版本的样本 - 提高密度、增加多样性、减小建筑块尺寸
        sustainable_sample = sample.copy()
        
        # 1. 提高建筑密度 - 使用90%分位数的值
        sustainable_sample['building_density'] = test_df['building_density'].quantile(0.9)
        
        # 2. 增加土地利用多样性 - 使用90%分位数的值
        sustainable_sample['land_use_diversity'] = test_df['land_use_diversity'].quantile(0.9)
        sustainable_sample['unique_land_uses'] = max(sample['unique_land_uses'] * 1.5, test_df['unique_land_uses'].quantile(0.9))
        
        # 3. 减小建筑块尺寸 - 使用20%分位数的值
        sustainable_sample['avg_building_size'] = test_df['avg_building_size'].quantile(0.45)
        sustainable_sample['avg_land_use_area_km2'] = test_df['avg_land_use_area_km2'].quantile(0.4)
        
        # 4. 增加道路网络复杂度和交叉口密度以支持更高密度的发展
        sustainable_sample['road_network_complexity'] = test_df['road_network_complexity'].quantile(0.85)
        sustainable_sample['intersection_density'] = test_df['intersection_density'].quantile(0.85)
        sustainable_sample['road_density_km'] = test_df['road_density_km'].quantile(0.85)
        
        # 5. 增加设施密度以支持更高密度的社区
        sustainable_sample['amenity_density'] = test_df['amenity_density'].quantile(0.9)
        
        # 生成可持续发展版本的提示
        sustainable_prompt = generate_enhanced_prompt(sustainable_sample, categorize_funcs)
        

        # 保存结果
        result_row = {
                'idx': sample['idx'],
                'test_type': 'sustainable_development',
                'prompt': sustainable_prompt,
                'original_building_density': sample['building_density'],
                'new_building_density': sustainable_sample['building_density'],
                'original_land_use_diversity': sample['land_use_diversity'],
                'new_land_use_diversity': sustainable_sample['land_use_diversity'],
                'original_avg_building_size': sample['avg_building_size'],
                'new_avg_building_size': sustainable_sample['avg_building_size'],
                'original_avg_land_use_area_km2': sample['avg_land_use_area_km2'],
                'new_avg_land_use_area_km2': sustainable_sample['avg_land_use_area_km2'],
                'original_unique_land_uses': sample['unique_land_uses'],
                'new_unique_land_uses': sustainable_sample['unique_land_uses']
        }
        sustainable_results.append(result_row)

    # 将结果转换为DataFrame并保存
    results_df = pd.DataFrame(sustainable_results)
    results_df.to_csv('./metrics_datagrido150/sustainable_development_prompts.csv', index=False)

    print(f"可持续发展测试完成，共生成 {len(results_df)} 个提示，已保存到 sustainable_development_prompts.csv")

    return results_df

# 执行可持续发展测试
if __name__ == "__main__":
    sustainable_results = run_sustainable_development_test()

    # 显示一些示例结果
    print("\n示例可持续发展测试结果:")
    for idx in [1199, 11959, 13085, 13674, 15219, 15814, 16417, 16717, 16892, 16984, 18099, 18659, 20624, 21684, 4659, 4918, 6850]:
        # 显示原始样本
        original_sample = sustainable_results[(sustainable_results['idx'] == idx) & 
                                             (sustainable_results['test_type'] == 'original')].iloc[0]
        print(f"\n原始样本ID: {original_sample['idx']}")
        print(f"原始提示: {original_sample['prompt'][:150]}...")
        
        # 显示可持续发展样本
        sustainable_sample = sustainable_results[(sustainable_results['idx'] == idx) & 
                                                (sustainable_results['test_type'] == 'sustainable_development')].iloc[0]
        print(f"\n可持续发展样本ID: {sustainable_sample['idx']}")
        print(f"原始建筑密度: {sustainable_sample['original_building_density']:.2f} -> 新建筑密度: {sustainable_sample['new_building_density']:.2f}")
        print(f"原始土地利用多样性: {sustainable_sample['original_land_use_diversity']:.2f} -> 新土地利用多样性: {sustainable_sample['new_land_use_diversity']:.2f}")
        print(f"原始建筑尺寸: {sustainable_sample['original_avg_building_size']:.2f} -> 新建筑尺寸: {sustainable_sample['new_avg_building_size']:.2f}")
        print(f"可持续发展提示: {sustainable_sample['prompt'][:150]}...")
