from pycollector.globals import isapi


class Label(object):
    def __init__(self, labeldict):
        pass

    
def _v1_applabel_to_piplabel(k=None):
    d = {"P002_P002C001_Closing": "person_closes_car_door",
         "P002_P002C001_Entering": "person_enters_car",
         "P002_P002C001_Exiting": "person_exits_car",
         "P002_P002C001_Opening": "person_opens_car_door",
         "P002_P002C002_Closing": "person_closes_car_trunk",
         "P002_P002C002_Loading": "person_loads_car",
         "P002_P002C002_Opening": "person_opens_car_trunk",
         "P002_P002C002_Unloading": "person_unloads_car",
         "P002_P002C003_Turning left": "car_turns_left",
         "P002_P002C003_Turning right": "car_turns_right",
         "P002_P002C003_U-turn": "car_makes_u_turn",
         "P002_P002C004_Picking up": "car_picks_up_person",
         "P002_P002C004_Starting": "car_starts",
         "P002_P002C004_Stopping": "car_stops",
         "P002_P002C005_Dropping off": "car_drops_off_person",
         "P002_P002C005_Starting": "car_starts",
         "P002_P002C005_Stopping": "car_stops",
         "P002_P002C006_Transferring": "person_transfers_object_to_car",
         "P002_P002C007_Reversing": "car_reverses",
         "P003_P003C001_Closing": "person_closes_motorcycle_door",  # remove me, meaningless
         "P003_P003C001_Entering": "person_enters_motorcycle",
         "P003_P003C001_Exiting": "person_exits_motorcycle",
         "P003_P003C001_Opening": "person_opens_motorcycle_door",  # remove me, meaningless
         "P003_P003C002_Closing": "person_closes_motorcycle_trunk",
         "P003_P003C002_Loading": "person_loads_motorcycle",
         "P003_P003C002_Opening": "person_opens_motorcycle_trunk",
         "P003_P003C002_Unloading": "person_unloads_motorcycle",
         "P003_P003C003_Turning left": "motorcycle_turns_left",
         "P003_P003C003_Turning right": "motorcycle_turns_right",
         "P003_P003C003_U-turn": "motorcycle_makes_u_turn",
         "P003_P003C004_Picking up": "motorcycle_picks_up_person",
         "P003_P003C004_Starting": "motorcycle_starts",
         "P003_P003C004_Stopping": "motorcycle_stops",
         "P003_P003C005_Dropping off": "motorcycle_drops_off_person",
         "P003_P003C005_Starting": "motorcycle_starts",
         "P003_P003C005_Stopping": "motorcycle_stops",
         "P003_P003C006_Transferring": "person_transfers_object_to_motorcycle",
         "P003_P003C007_Reversing": "motorcycle_reverses",
         "P004_P004C001_Closing": "person_closes_facility_door",
         "P004_P004C001_Entering": "person_enters_scene_through_structure",
         "P004_P004C001_Exiting": "person_exits_scene_through_structure",
         "P004_P004C001_Opening": "person_opens_facility_door",
         "P004_P004C002_Entering": "person_enters_scene_through_structure",
         "P004_P004C002_Exiting": "person_exits_scene_through_structure",
         "P004_P004C003_Reading": "person_reads_document",
         "P004_P004C003_Sitting": "person_sits_down",
         "P004_P004C003_Standing": "person_stands_up",
         "P004_P004C004_Dropping off": "person_puts_down_object",
         "P004_P004C004_Picking up": "person_picks_up_object",
         "P004_P004C005_Abandoning": "person_abandons_package",
         "P004_P004C005_Sets down": "person_puts_down_object",
         "P004_P004C006_Carrying": "person_carries_heavy_object",
         "P004_P004C007_Talking on": "person_talks_on_phone",
         "P004_P004C007_Texting on": "person_texts_on_phone",
         "P004_P004C008_Sitting": "person_sits_down",
         "P004_P004C008_Standing": "person_stands_up",
         "P004_P004C008_Using laptop": "person_interacts_with_laptop",
         "P004_P004C009_Purchasing": "person_purchases_from_machine",
         "P004_P004C010_Dropping off": "person_puts_down_object",
         "P004_P004C010_Picking up": "person_picks_up_object",
         "P004_P004C010_Riding": "person_rides_bicycle",
         "P005_P005C001_Talking to": "person_talks_to_person",
         "P005_P005C001_Touching hands": "hand_interacts_with_person_highfive",
         "P005_P005C002_Stealing": "person_steals_object",
         "P005_P005C003_Purchasing": "person_purchases_from_cashier",  # Remove me
         "P005_P005C004_Transferring": "person_transfers_object_to_person",
         "P005_P005C005_Hugging": "person_embraces_person",
         "P005_P005C005_Touching hands": "hand_interacts_with_person_holdhands",
         "P005_P005C006_Talking to": "person_talks_to_person",
         "P005_P005C006_Touching hands": "hand_interacts_with_person_shakehands",
    }
    return d[k] if k is not None else d


def piplabel_to_mevalabel():
    d = {v: v for (k, v) in _v1_applabel_to_piplabel().items()}

    d = {k: "hand_interacts_with_person" if "hand" in v else v for (k, v) in d.items()}
    d = {
        k: v.replace("car", "vehicle")
        if (("_car" in v or "car_" in v) and ("carries" not in v))
        else v
        for (k, v) in d.items()
    }
    d = {
        k: v.replace("motorcycle", "vehicle") if "motorcycle" in v else v
        for (k, v) in d.items()
    }
    d = {
        k: v.replace("vehicle_trunk", "trunk") if "vehicle_trunk" in v else v
        for (k, v) in d.items()
    }
    d["person_transfers_object_to_car"] = "person_transfers_object"
    d["person_transfers_object_to_motorcycle"] = "person_transfers_object"
    d["person_transfers_object_to_person"] = "person_transfers_object"
    d["person_purchases_from_machine"] = "person_purchases"
    d["person_purchases_from_cashier"] = "person_purchases"
    d = {
        k: v
        for (k, v) in d.items()
        if k
        not in set(["person_closes_motorcycle_door", "person_opens_motorcycle_door"])
    }
    return d


def mevalabel_to_index():

    return {
        "hand_interacts_with_person": 0,
        "person_abandons_package": 1,
        "person_carries_heavy_object": 2,
        "person_closes_facility_door": 3,
        "person_closes_trunk": 4,
        "person_closes_vehicle_door": 5,
        "person_embraces_person": 6,
        "person_enters_scene_through_structure": 7,
        "person_enters_vehicle": 8,
        "person_exits_scene_through_structure": 9,
        "person_exits_vehicle": 10,
        "person_interacts_with_laptop": 11,
        "person_loads_vehicle": 12,
        "person_opens_facility_door": 13,
        "person_opens_trunk": 14,
        "person_opens_vehicle_door": 15,
        "person_picks_up_object": 16,
        "person_purchases": 17,
        "person_puts_down_object": 18,
        "person_reads_document": 19,
        "person_rides_bicycle": 20,
        "person_sits_down": 21,
        "person_stands_up": 22,
        "person_steals_object": 23,
        "person_talks_on_phone": 24,
        "person_talks_to_person": 25,
        "person_texts_on_phone": 26,
        "person_transfers_object": 27,
        "person_unloads_vehicle": 28,
        "vehicle_drops_off_person": 29,
        "vehicle_makes_u_turn": 30,
        "vehicle_picks_up_person": 31,
        "vehicle_reverses": 32,
        "vehicle_starts": 33,
        "vehicle_stops": 34,
        "vehicle_turns_left": 35,
        "vehicle_turns_right": 36,
    }


def piplabel_to_index():
    return {
        "car_drops_off_person": 0,
        "car_makes_u_turn": 1,
        "car_picks_up_person": 2,
        "car_reverses": 3,
        "car_starts": 4,
        "car_stops": 5,
        "car_turns_left": 6,
        "car_turns_right": 7,
        "hand_interacts_with_person_highfive": 8,
        "hand_interacts_with_person_holdhands": 9,
        "hand_interacts_with_person_shakehands": 10,
        "motorcycle_drops_off_person": 11,
        "motorcycle_makes_u_turn": 12,
        "motorcycle_picks_up_person": 13,
        "motorcycle_reverses": 14,
        "motorcycle_starts": 15,
        "motorcycle_stops": 16,
        "motorcycle_turns_left": 17,
        "motorcycle_turns_right": 18,
        "person_abandons_package": 19,
        "person_carries_heavy_object": 20,
        "person_closes_car_door": 21,
        "person_closes_car_trunk": 22,
        "person_closes_facility_door": 23,
        "person_closes_motorcycle_trunk": 24,
        "person_embraces_person": 25,
        "person_enters_car": 26,
        "person_enters_scene_through_structure": 27,
        "person_exits_car": 28,
        "person_exits_scene_through_structure": 29,
        "person_interacts_with_laptop": 30,
        "person_loads_car": 31,
        "person_loads_motorcycle": 32,
        "person_opens_car_door": 33,
        "person_opens_car_trunk": 34,
        "person_opens_facility_door": 35,
        "person_opens_motorcycle_trunk": 36,
        "person_picks_up_object": 37,
        "person_purchases_from_machine": 38,
        "person_puts_down_object": 39,
        "person_reads_document": 40,
        "person_rides_bicycle": 41,
        "person_sits_down": 42,
        "person_stands_up": 43,
        "person_steals_object": 44,
        "person_talks_on_phone": 45,
        "person_talks_to_person": 46,
        "person_texts_on_phone": 47,
        "person_transfers_object_to_car": 48,
        "person_transfers_object_to_motorcycle": 49,
        "person_transfers_object_to_person": 50,
        "person_unloads_car": 51,
        "person_unloads_motorcycle": 52,
    }


#def _applabel_to_longlabel():
#    """FIXME: this mapping should be baked into the app"""
#    assert isapi('v1')
#    print(
#        '[collector.dataset.applabel_to_longlabel]:  Scanning table "co_Activities_Mobile_ID_Dict_Dev"'
#    )
#    t = collector.admin.Backend()._dynamodb_resource.Table(
#        "co_Activities_Mobile_ID_Dict_Dev"
#    )
#    return {d["Mobile_ID"]: d["Instance_ID"] for d in t.scan()["Items"]}


