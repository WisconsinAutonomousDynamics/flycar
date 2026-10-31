"""Perturb a loaded scenario so drivers can be tested on variations of it.

    spec = load_spec("spec.yaml")
    variant, drawn = randomize(load_scenario("scenarios/light_seek.yaml"), spec, seed)
    env = FlyCarEnv(variant)

A spec maps a dotted path to a [low, high] range, e.g.
    lighting.ambient:     [0.0, 1.0]
    lights[*].intensity:  [0.5, 2.0]      # [*] = every item of a list section
    eyes.resolution:      [8, 32]         # [rows, cols] list -> one square size
"""
import copy
import random
import re
import yaml
WILDCARD = re.compile(r"^(?P<section>[\w.]+)\[\*\]\.(?P<key>[\w.]+)$")

# Reads values from files and checks if its a valid range
def load_spec(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    spec = {}
    for name, rng in raw.items():
        if not (isinstance(rng, list) and len(rng) == 2):
            raise ValueError(f"{name}: range must be [low, high], got {rng!r}")
        low, high = rng
        if low > high:
            raise ValueError(f"{name}: low {low} is greater than high {high}")
        spec[name] = (low, high)
    return spec

# finds the correct value once read
def find(node, dotted):
    *parents, leaf = dotted.split(".")
    for part in parents:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"no such key '{dotted}'")
        node = node[part]
    if not isinstance(node, dict) or leaf not in node:
        raise KeyError(f"no such key '{dotted}'")
    return node, leaf

# makes a random value for the scenario
def draw(rng, low, high, current):
    if isinstance(current, list):            
        value = draw(rng, low, high, current[0])
        return [value for _ in current]      
    if type(current) is int and isinstance(low, int) and isinstance(high, int):
        return rng.randint(low, high)        
    return rng.uniform(low, high)            

# finds and randomizes each value in spec.yaml
def randomize(base, spec, seed):
    from flycar.env.scenario import load_scenario  

    rng = random.Random(f"flycar-randomize/{seed}")  
    variant = copy.deepcopy(base)
    drawn = {}

    for name in sorted(spec):                
        low, high = spec[name]
        match = WILDCARD.match(name)
        if match:                           
            holder, key = find(variant, match["section"])
            for i, item in enumerate(holder[key]):
                parent, leaf = find(item, match["key"])
                parent[leaf] = draw(rng, low, high, parent[leaf])
                drawn[f"{match['section']}[{i}].{match['key']}"] = parent[leaf]
        else:                                
            parent, leaf = find(variant, name)
            parent[leaf] = draw(rng, low, high, parent[leaf])
            drawn[name] = parent[leaf]

    return load_scenario(variant), drawn     


if __name__ == "__main__":
    import sys
    from flycar.env.scenario import load_scenario

    scenario_file, spec_file, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    _, drawn = randomize(load_scenario(scenario_file), load_spec(spec_file), seed)
    for name, value in drawn.items():
        print(f"{name} = {value}")

