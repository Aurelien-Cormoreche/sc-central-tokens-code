from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
import argparse
import json
from tqdm import tqdm

def extract_polygons(geom):
    """Extract Polygon/MultiPolygon from any geometry type"""
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    elif isinstance(geom, GeometryCollection):
        # Extract only Polygon and MultiPolygon from the collection
        polygons = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if len(polygons) == 0:
            return None
        elif len(polygons) == 1:
            return polygons[0]
        else:
            # Flatten all polygons into a single MultiPolygon
            all_polys = []
            for p in polygons:
                if isinstance(p, Polygon):
                    all_polys.append(p)
                else:  # MultiPolygon
                    all_polys.extend(list(p.geoms))
            return MultiPolygon(all_polys)
    else:
        return None

def fix_geojson(input_file_path, output_file_path):
    print("--- Reading Input File ---")
    with open(input_file_path, 'r') as f_in:
        geojson_data = json.load(f_in)
    print(f"--- Finished Reading {len(geojson_data)} features ---")
    
    fixed_features = []
    skipped = 0
    
    for cell_type in tqdm(geojson_data):
        geom_shape = shape(cell_type['geometry'])
        
        # Fix invalid geometries
        if not geom_shape.is_valid:
            geom_shape = make_valid(geom_shape)
        
        # Extract only polygons (in case make_valid created a GeometryCollection)
        geom_shape = extract_polygons(geom_shape)
        
        if geom_shape is None or geom_shape.is_empty:
            skipped += 1
            continue
        
        # Simplify
        #geom_shape = geom_shape.simplify(0.5, preserve_topology=True)
        
        # Update geometry
        cell_type['geometry'] = mapping(geom_shape)
        fixed_features.append(cell_type)
    
    print(f"--- Processed {len(fixed_features)} valid features, skipped {skipped} ---")
    print("--- Writing To Output File ---")    
    with open(output_file_path, 'w') as f_out:
        json.dump(fixed_features, f_out)
    
    print(f"--- Done! Output saved to {output_file_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=str, help='path to the raw CellViT cells.geojson')
    parser.add_argument('output', type=str, help='path to write the fixed geojson to')
    args = parser.parse_args()

    fix_geojson(args.input, args.output)