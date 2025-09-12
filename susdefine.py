import pandas as pd
import numpy as np
import random

def create_categorization_functions(df):
    """
    Creates a set of functions to categorize urban metrics based on their percentile
    distribution within a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing urban metrics.

    Returns:
        dict: A dictionary of categorization functions, with metric names as keys.
    """
    # Dictionary to store percentile thresholds for each metric
    thresholds = {}
    
    # List of metrics to categorize
    metrics = [
        'building_density', 'avg_building_size',
        'road_density_km', 'intersection_density', 'avg_land_use_area_km2',
        'road_network_complexity', 'land_use_diversity', 'unique_land_uses',
        'amenity_density'
    ]
    
    # Calculate percentile thresholds (40th, 60th, 80th, 95th) for each metric
    for metric in metrics:
        thresholds[metric] = [
            df[metric].quantile(0.4),
            df[metric].quantile(0.6),
            df[metric].quantile(0.8),
            df[metric].quantile(0.95)
        ]
    
    # --- Define categorization functions for each metric ---
    
    def categorize_building_density(value):
        """Categorizes building density into 'very low' to 'very high'."""
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
        """Categorizes average building size."""
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
        """Categorizes road density."""
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
        """Categorizes intersection density."""
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
        """Categorizes average land use area."""
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
        """Categorizes road network complexity."""
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
        """Categorizes land use diversity."""
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
        """
        Categorizes the number of unique land uses. Note: This function uses a
        slightly different thresholding logic due to the data's distribution.
        """
        t = thresholds['unique_land_uses']
        if value < t[2]:  # Less than the 80th percentile
            return "minimal", "almost single-purpose area"
        elif value < t[3]:  # Less than the 95th percentile
            return "several", "several different land uses"
        elif value < t[3] * 2:  # Up to double the 95th percentile
            return "many", "many different land uses creating diversity"
        else:
            return "extensive", "extensive variety of land uses creating a highly mixed environment"
        
    def categorize_amenity_density(value):
        """
        Categorizes amenity density with a special case for zero values, since
        the data is heavily skewed towards areas with no amenities.
        """
        t = thresholds['amenity_density']
        if value == 0:
            return "none", "no amenities or services available"
        elif value <= t[3] / 4:  # Up to 25% of the 95th percentile value
            return "very low", "minimal amenities with only basic services"
        elif value <= t[3] / 2:  # Up to 50% of the 95th percentile value
            return "low", "limited amenities and services"
        elif value <= t[3]:  # Up to the 95th percentile
            return "medium", "moderate number of amenities"
        elif value <= t[3] * 2:  # Up to 2x the 95th percentile
            return "high", "abundant amenities and services"
        else:  # Above 2x the 95th percentile
            return "very high", "extremely high concentration of amenities and services"
    
    # Return all categorization functions in a dictionary
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
    """
    Generates a detailed, human-readable text prompt for an image generation model,
    combining numerical data with descriptive categories.

    Args:
        row (pd.Series): A row from the DataFrame representing a single urban area.
        categorize_funcs (dict): A dictionary of categorization functions.

    Returns:
        str: The generated text prompt.
    """
    # Get categories and descriptions for all metrics
    bd_cat, bd_desc = categorize_funcs['building_density'](row['building_density'])
    bs_cat, bs_desc = categorize_funcs['avg_building_size'](row['avg_building_size'])
    rd_cat, rd_desc = categorize_funcs['road_density_km'](row['road_density_km'])
    id_cat, id_desc = categorize_funcs['intersection_density'](row['intersection_density'])
    lua_cat, lua_desc = categorize_funcs['avg_land_use_area_km2'](row['avg_land_use_area_km2'])
    rnc_cat, rnc_desc = categorize_funcs['road_network_complexity'](row['road_network_complexity'])
    lud_cat, lud_desc = categorize_funcs['land_use_diversity'](row['land_use_diversity'])
    ulu_cat, ulu_desc = categorize_funcs['unique_land_uses'](row['unique_land_uses'])
    ad_cat, ad_desc = categorize_funcs['amenity_density'](row['amenity_density'])
    
    # --- Structure the prompt into logical sections (Density, Diversity, Design) ---
    
    # DENSITY section: includes building, road, intersection, and amenity density
    density_section = (
        f"This area has {bd_cat} building density ({row['building_density']:.1f} buildings per square kilometer), {bd_desc}. "
        f"It features {rd_cat} road density ({row['road_density_km']:.1f} km per square kilometer) where {rd_desc}, "
        f"with {id_cat} intersection density ({row['intersection_density']:.1f} intersections per square kilometer) where {id_desc}. "
        f"The area has {ad_cat} amenity density ({row['amenity_density']:.3f} amenities per square kilometer) where {ad_desc}. "
    )
    
    # DIVERSITY section: includes land use diversity and number of unique land uses
    diversity_section = (
        f"The neighborhood exhibits {lud_cat} land use diversity (index: {row['land_use_diversity']:.2f}) where {lud_desc}, "
        f"containing {ulu_cat} unique land uses ({row['unique_land_uses']}) where {ulu_desc}. "
    )
    
    # DESIGN section: includes building size, road network complexity, and land use area size
    design_section = (
        f"The urban design features {bs_cat} buildings (averaging {row['avg_building_size']:.1f} square meters), {bs_desc}. "
        f"The road network shows {rnc_cat} complexity ({row['road_network_complexity']:.2f}) where {rnc_desc}. "
        f"The land use areas are {lua_cat} (averaging {row['avg_land_use_area_km2']:.3f} square kilometers) where {lua_desc}. "
    )
    
    # Combine all sections into the final prompt
    prompt = (
        f"A realistic image of an urban area in Orlando--left half as satellite image, right half as corresponding land use map. "
        f"{density_section}{diversity_section}{design_section}"
        f"The image should be viewed from above, in high resolution aerial photography style, "
        f"clearly showing the urban morphology and land use patterns described."
    )
    
    return prompt

def run_sustainable_development_test():
    """
    Executes a test by generating prompts for 'sustainable development' scenarios.
    This involves modifying original data points to reflect characteristics of
    more sustainable urban areas (e.g., higher density, more diversity).
    """
    # Read the test dataset from a specified CSV file
    try:
        test_df = pd.read_csv('./metrics_datagrido150/test_set_below_0.7.csv')
    except FileNotFoundError:
        print("Error: The file 'test_set_below_0.7.csv' was not found.")
        return None

    # Select specific indices to test from the DataFrame
    target_indices = [
        1199, 11959, 13085, 13674, 15219, 15814, 16417, 16717, 16892,
        16984, 18099, 18659, 20624, 21684, 4659, 4918, 6850
    ]
    test_samples = test_df[test_df['idx'].isin(target_indices)].copy()

    # Create the categorization functions based on the entire test set
    categorize_funcs = create_categorization_functions(test_df)

    # List to store the results of the prompt generation
    sustainable_results = []

    # Iterate through each selected sample to generate original and sustainable prompts
    for _, sample in test_samples.iterrows():
        # Generate the prompt for the original, unmodified sample
        original_prompt = generate_enhanced_prompt(sample, categorize_funcs)
        
        result_row_original = {
            'idx': sample['idx'],
            'test_type': 'original',
            'prompt': original_prompt
        }
        sustainable_results.append(result_row_original)
        
        # Create a copy to generate the 'sustainable development' version
        sustainable_sample = sample.copy()
        
        # --- Modify metric values to simulate a sustainable development scenario ---
        
        # 1. Increase building density to the 90th percentile
        sustainable_sample['building_density'] = test_df['building_density'].quantile(0.9)
        
        # 2. Increase land use diversity and the number of unique land uses
        sustainable_sample['land_use_diversity'] = test_df['land_use_diversity'].quantile(0.9)
        sustainable_sample['unique_land_uses'] = max(
            sample['unique_land_uses'] * 1.5,
            test_df['unique_land_uses'].quantile(0.9)
        )
        
        # 3. Decrease average building size and land use area to promote smaller-scale development
        sustainable_sample['avg_building_size'] = test_df['avg_building_size'].quantile(0.45)
        sustainable_sample['avg_land_use_area_km2'] = test_df['avg_land_use_area_km2'].quantile(0.4)
        
        # 4. Increase road network complexity, density, and intersection density to support higher density
        sustainable_sample['road_network_complexity'] = test_df['road_network_complexity'].quantile(0.85)
        sustainable_sample['intersection_density'] = test_df['intersection_density'].quantile(0.85)
        sustainable_sample['road_density_km'] = test_df['road_density_km'].quantile(0.85)
        
        # 5. Increase amenity density to support the needs of a more densely populated community
        sustainable_sample['amenity_density'] = test_df['amenity_density'].quantile(0.9)
        
        # Generate the prompt for the modified, 'sustainable' sample
        sustainable_prompt = generate_enhanced_prompt(sustainable_sample, categorize_funcs)
        
        # Store the results, including original and new metric values for comparison
        result_row_sustainable = {
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
        sustainable_results.append(result_row_sustainable)

    # Convert the list of results to a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(sustainable_results)
    results_df.to_csv('./metrics_datagrido150/sustainable_development_prompts.csv', index=False)

    print(f"Sustainable development test completed. Generated {len(results_df)} prompts and saved to sustainable_development_prompts.csv")

    return results_df

# --- Script Execution ---
if __name__ == "__main__":
    sustainable_results = run_sustainable_development_test()

    if sustainable_results is not None:
        print("\nExample Sustainable Development Test Results:")
        # Display examples
        target_indices = [
            1199,3344
        ]#choose from your own idx
        for idx in target_indices:
            # Display the original sample's prompt and metrics
            original_sample = sustainable_results[
                (sustainable_results['idx'] == idx) &
                (sustainable_results['test_type'] == 'original')
            ].iloc[0]
            print(f"\nOriginal Sample ID: {original_sample['idx']}")
            print(f"Original Prompt: {original_sample['prompt'][:150]}...")
            
            # Display the sustainable sample's prompt and new metrics
            sustainable_sample = sustainable_results[
                (sustainable_results['idx'] == idx) &
                (sustainable_results['test_type'] == 'sustainable_development')
            ].iloc[0]
            print(f"\nSustainable Development Sample ID: {sustainable_sample['idx']}")
            print(f"Original Building Density: {sustainable_sample['original_building_density']:.2f} -> New Building Density: {sustainable_sample['new_building_density']:.2f}")
            print(f"Original Land Use Diversity: {sustainable_sample['original_land_use_diversity']:.2f} -> New Land Use Diversity: {sustainable_sample['new_land_use_diversity']:.2f}")
            print(f"Original Building Size: {sustainable_sample['original_avg_building_size']:.2f} -> New Building Size: {sustainable_sample['new_avg_building_size']:.2f}")
            print(f"Sustainable Development Prompt: {sustainable_sample['prompt'][:150]}...")
