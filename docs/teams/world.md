# Team: World (`world`)

This page belongs to the team. Keep it up to date here instead of in the README.
The Phase 1 work guide (tasks, parts A to C) is [`teams/team-world.md`](../../teams/team-world.md).

## What you own

| Path | What it is |
|---|---|
| `flycar/env/world.py` | `sample_layout()` (rolls the dice) and `World` (walls, obstacles, lamps) |
| `scenarios/` | Every scenario file **except** the frozen baselines `empty_arena.yaml` and `light_seek.yaml` and the other teams' `team_*.yaml` scratch files |
| `flycar/config/world.py` | Scenario sections `arena`, `lighting`, `lights`, `obstacles`, `motion`: defaults and checks |
| `tests/test_world.py` | Your tests |
| `scenarios/team_world.yaml` | Your scratch scenario |
| `flycar/drivers/contrib/world_driver.py` | Your capstone driver (observation only, no pychrono) |
| `teams/team-world.md` | Your Phase 1 work guide |
| `docs/teams/world.md` | This page |

## Contracts you must not break

Changing these needs a lead-approved `api-change` issue (see
[`docs/CONTRACTS.md`](../CONTRACTS.md)):

- **The `layout` dict** returned by `sample_layout()` (contract 2): every key
  and type. The logger, replay, dashboard and eyes all read it.
- **`World.current_lights()`** and **`World.layout`** (contract 4). The eyes
  ask `current_lights()` every time they take a picture, so moving lamps only
  needs a change here.
- **The tag scheme** (contract 4): walls and obstacles get tags
  `>= WORLD_TAG_BASE`; the contact sensor relies on it.
- **The step order in `core._physics_step()`** (contract 7):
  `World.synchronize(time)` / `World.advance(step)` are where moving things
  will be updated.

## Your reserved config

```yaml
motion:
  obstacles: []   # planned: moving obstacles
  lights: []      # planned: moving lights
```

Anything but empty lists is rejected with "not implemented yet" until you
implement it.

## Things to know

- `empty_arena.yaml` and `light_seek.yaml` are frozen: the baseline test
  replays them. Make a new file for a new experiment.
- To expose debug data, add `extra_info()` to `World`; it shows up in
  `info["extras"]["world"]`.

## Running your tests

```
pytest tests/test_world.py
python run.py --scenario scenarios/team_world.yaml --driver braitenberg --dashboard
```

Scenario Tests w/ seeds:

corridor:
  SUCCESSES: 0
  FAILS: 20
  RES...SUCCESS: 0,  FAIL: 20
  Seeds: [7949528353460712003, 48643669649390084560, 38822640449874025800, 67190123916966107326, 4964573488322863148, 65277150694262131886, 67083864156562711423, 77321216337623260117, 37540881235646746198, 86324948261384942947, 70451366582945213552, 25217293363813595496, 24196311666654700997, 63697827618881996672, 77131937038794669313, 4521375323121617036, 37546880297253473286, 29120802665598657177, 56624679705478023252, 10794288704509655155]

two_lamps:
  RES...SUCCESS: 1,  FAIL: 19
  Seeds: [73224548809307847603, 59481584109989199535, 60390752743642349686, 93399364603694839884, 41534261868854141272, 42300323904421465381, 61902135562134848250, 42019488306643972022, 53453804751389987774, 89009997760069474470, 13765399424896945706, 88887325627267064363, 22941749174519855042, 12885055406330373335, 8598238763576567852, 18792915731329349345, 96106342600393856160, 76004319602869277233, 33513026973557410037, 39405356281167048806]

occluded_lamp:
  RES...SUCCESS: 9,  FAIL: 11
  Seeds: [62910360416998952119, 13901379358849050977, 64860206599970758661, 60551314608478344143, 42444746228406214389, 20602403594770576518, 82405139848671685721, 12927441136970061725, 43181366078679406664, 34309925090836567171, 98724081867312666540, 83931968650245023141, 80549831778906539185, 24072475650594855679, 28098629031018544936, 99817687753868104186, 17497725560347556897, 11209803968285040943, 64710658312255349734, 83138362234584111641]

cluttered:
  RES...SUCCESS: 9,  FAIL: 11
  Seeds: [62910360416998952119, 13901379358849050977, 64860206599970758661, 60551314608478344143, 42444746228406214389, 20602403594770576518, 82405139848671685721, 12927441136970061725, 43181366078679406664, 34309925090836567171, 98724081867312666540, 83931968650245023141, 80549831778906539185, 24072475650594855679, 28098629031018544936, 99817687753868104186, 17497725560347556897, 11209803968285040943, 64710658312255349734, 83138362234584111641]

dim_lamp:
  RES...SUCCESS: 11,  FAIL: 9
  Seeds: [15383786065792415577, 72012092152500586178, 19999163541961095132, 45778545056261784237, 91900587925595696689, 62339302575946932460, 62947125035308050441, 43708804750527472676, 67511279219608040421, 87463514787098796517, 10250830044544326709, 1741051502530426893, 43618732370750422425, 91174352354610881931, 90222214157404977183, 91835261656430121184, 81375554585207587837, 93720747624463128514, 29734373424248514336, 83248438945081251112]