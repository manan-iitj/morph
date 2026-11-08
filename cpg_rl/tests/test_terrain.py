import sys
sys.path.insert(0, '.')
from terrain_generator import TerrainConfig, TerrainGenerator

gen = TerrainGenerator(TerrainConfig(terrain_type='stairs', difficulty='medium', random_seed=42))
gen.save_to_file('stairs.xml')

query_fn, obstacle_fn = gen.get_terrain_query()
print(f'Stairs XML: {len(gen.generate_xml())} chars')
print(f'Stairs: {gen.get_metadata()["n_stairs"]}')
print(f'Height at (2.0, 0.0): {query_fn(2.0, 0.0):.3f} m')
print(f'Height at (3.0, 0.0): {query_fn(3.0, 0.0):.3f} m')
print(f'Obstacle at (2.0, 0.0): {obstacle_fn(2.0, 0.0)}')
print(f'Obstacle at (3.0, 0.0): {obstacle_fn(3.0, 0.0)}')
PYEOF