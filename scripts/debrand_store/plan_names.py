#!/usr/bin/env python3
"""Produce the de-branding rename/label plan for an exported store package.

Categories are a hand-verified table, not filename inference: the albedo atlases
were inspected as contact sheets and a third of them carry a brand the filename
does not mention (``bean_tin`` is Heinz, ``soda_can_3`` is Dr Pepper,
``shampoo1`` is Palmolive, ``scan_3d_mau_chai_vim`` is Cif).

Emits ``rename_map.yaml``: per product a generic asset id, a category, a package
form, a synthetic wordmark and a display name.  Assets whose *silhouette* is
itself protected are flagged ``shape_ip`` and given a bbox-matched replacement
drawn from the surviving products, so the shelves stay full.

Usage:
    python scripts/debrand_store/plan_names.py conf/debrand/inventory.json \
        --out conf/debrand/rename_map.yaml
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

# Verified by inspecting every product's primary albedo atlas.
CATEGORY = {
    "350_jar_rotated": "canned", "3d_juice_box_rotated": "juice",
    "43b88b2c6fa6462aa4c9d2d3c5e062a0_rotated": "condiment", "ace": "juice",
    "acidophilic_milk_rotated": "dairy", "apple_juice_rotated": "juice",
    "axe_deodorant_rotated": "deodorant", "barbasol_shaving_cream_can_rotated": "shaving",
    "bean_tin_rotated": "canned", "beans_rotated": "canned", "beef_can_rotated": "canned",
    "bom_ba_milk_rotated": "dairy", "bottle_of_detergent": "detergent",
    "bottle_of_soap": "soap", "brazilian_milk_bottle": "dairy",
    "can_of_sweet_corn": "canned", "canarian_cafe-_jar_of_olives_rotated": "canned",
    "canned_energy_drink_art-soul_rotated": "energy", "canned_food_rotated": "canned",
    "carbonara_jar_rotated": "sauce", "carpet_cleaner_1200ml": "cleaner",
    "cepita_peach_juice_bottle": "juice", "cleaning_spray_rotated": "cleaner",
    "coca-_cola_zero_sugar_byte_no_base_rotated": "cola", "coca-coka_can_rotated": "cola",
    "cocaColaCan": "cola", "coca_cola_bottle_rotated": "cola",
    "cosmetics__skin_care_product_-_free_rotated": "skincare", "dish_detergent": "detergent",
    "dos_pinos_-_jugo_verde_detox_rotated": "juice",
    "dos_pinos_-_leche_descremada_0_rotated": "dairy", "elmers_glue_shampoo_rotated": "haircare",
    "energy_drink_can_free_model_rotated": "energy",
    "energy_drink_game_ready_model_rotated": "energy",
    "face_eye_moisturizer_cream_rotated": "skincare",
    "face_eyes_moisturizer_cream_rotated": "skincare",
    "fallout_4_nuka_cola_bottle_rotated": "soda", "fanta_can_1_rotated": "soda",
    "fanta_naranja_2l_rotated": "soda", "food_can_415g_rotated": "canned",
    "gameready_lowpoly_juice_bottle_free_download_rotated": "juice",
    "gfuel__spiderman_edition__gameready__energy_rotated": "energy",
    "glass_bottle_3d_scan_rotated": "soda", "goop_tin_rotated": "canned",
    "grape_juice_1": "juice", "grape_juice_rotated": "juice", "green_beans_rotated": "canned",
    "guoba_chili_energy_drink_rotated": "energy", "hand_cream_rotated": "skincare",
    "hellmanns_hot_sauce_rotated": "sauce", "hot_sauce_bottle_rotated": "sauce",
    "juice_box_1_rotated": "juice", "juice_box_rotated": "juice", "juicebox00": "juice",
    "kendu_energy_drink_can_rotated": "energy", "ketchup_aridll_rotated": "condiment",
    "ketchup_rotated": "condiment", "kitchen_spray_rotated": "cleaner",
    "lunch_canned_luncheon_meat_rotated": "canned", "milkHandle": "dairy",
    "milk_2_rotated": "dairy", "milk_box_french_rotated": "dairy", "milk_carton": "dairy",
    "milk_carton_1": "dairy", "milk_of_human_kindness_rotated": "dairy",
    "milk_test_rotated": "dairy", "monster_energy_drink_rotated": "energy",
    "monster_energy_pipeline_punch_355ml_jpn_rotated": "energy",
    "monster_ultra_white_rotated": "energy", "mountainDew": "soda",
    "nivea_-_refreshing_day_cream_-_by_smakologg_rotated": "skincare",
    "nivea_aftershave_rotated": "shaving", "nivea_man_shampoo": "haircare",
    "non_stop_energy_rotated": "energy", "old_spice_deodorant_lowpoly_rotated": "deodorant",
    "old_spice_rotated": "deodorant", "oldspice_deodorant_stick_rotated": "deodorant",
    "orange_juice_bottle_rotated": "juice", "orange_juice_rotated": "juice",
    "parquet_detergent_with_almond_oil_1220ml": "cleaner", "pepsi": "cola",
    "pepsi_max_coke_can__pepsi_max_kola_kutusu_rotated": "cola",
    "perwoll_detergent_3d_model": "detergent", "prime_-_lemon_lime_rotated": "energy",
    "prime_-_tropical_punch_rotated": "energy", "redbull_1_rotated": "energy",
    "redbull__green_energy_drink_can_rotated": "energy", "room_spray_rotated": "cleaner",
    "royal_goyave_rose_bio_1l_rotated": "juice", "scan_3d_mau_chai_vim_rotated": "cleaner",
    "shampoo1_rotated": "haircare", "shampoo_grandes_heroes_rotated": "haircare",
    "shock_energy_drink": "energy", "soda_can_2_rotated": "soda", "soda_can_3_rotated": "soda",
    "soy_sauce_1_rotated": "sauce", "spray_crema_nivea_50_rotated": "skincare",
    "sprite2p25l": "soda", "sprite_bottle_custom_rotated": "soda",
    "super_awesome_milk_carton": "dairy", "sweet_baby_rays_bbq_sauce_rotated": "sauce",
    "tabasco_sauce_rotated": "sauce", "the_perfect_juice_rotated": "juice",
    "thick_vegetable_charge_juice_rotated": "juice",
    "tide_detergent_bottle_3d_model_fanmade": "detergent", "tomato_paste_tin": "canned",
    "topo_chico_mineral_water_rotated": "water", "tuna_can": "canned",
    "vintage_canned_corn_1": "canned",
    "yasai_seikatsu_100_berry_salad_juice_rotated": "juice",
}

# Silhouette is itself the mark, or the mesh is third-party fan IP.  Retexturing
# does not fix these, so they are dropped and their item prims re-pointed.
SHAPE_IP = {
    "fallout_4_nuka_cola_bottle_rotated": "Fallout / Nuka-Cola prop; distinctive ribbed bottle",
    "gfuel__spiderman_edition__gameready__energy_rotated": "G-Fuel x Marvel Spider-Man edition",
    "coca_cola_bottle_rotated": "Coca-Cola contour bottle is a registered shape mark",
    "topo_chico_mineral_water_rotated": "distinctive embossed Topo Chico bottle",
}

# Package form.  Matched on *tokens*, not substrings: "scan" contains "can" and
# "cola" used to match a can hint, which mislabelled glass bottles as cans.
FORM_TOKENS = [
    ("carton", {"carton", "box", "juicebox", "juicebox00"}),
    ("jar", {"jar", "olives"}),
    ("spray", {"spray"}),
    ("tube", {"cream", "moisturizer"}),
    ("stick", {"stick"}),
    ("bottle", {"bottle"}),
    ("tin", {"tin"}),
    ("can", {"can"}),
]
FORM_OVERRIDE = {
    "sprite2p25l": "bottle", "fanta_naranja_2l_rotated": "bottle", "ace": "bottle",
    "43b88b2c6fa6462aa4c9d2d3c5e062a0_rotated": "bottle", "milkHandle": "bottle",
    "elmers_glue_shampoo_rotated": "bottle", "nivea_man_shampoo": "bottle",
    "grape_juice_1": "carton", "cocaColaCan": "can", "pepsi": "can",
}
FORM_DEFAULT = {
    "cola": "can", "soda": "can", "energy": "can", "juice": "carton", "dairy": "carton",
    "water": "bottle", "canned": "tin", "sauce": "bottle", "condiment": "bottle",
    "haircare": "bottle", "skincare": "tube", "deodorant": "stick", "shaving": "can",
    "detergent": "bottle", "cleaner": "bottle", "soap": "bottle",
}

DISPLAY = {
    "cola": "Cola", "soda": "Sparkling Drink", "energy": "Energy Drink", "juice": "Juice",
    "dairy": "Milk", "water": "Mineral Water", "canned": "Canned Food", "sauce": "Sauce",
    "condiment": "Condiment", "haircare": "Shampoo", "skincare": "Skin Cream",
    "deodorant": "Deodorant", "shaving": "Shaving Foam", "detergent": "Detergent",
    "cleaner": "Cleaning Spray", "soap": "Hand Soap",
}

# Synthetic wordmarks: pronounceable, clearly invented, deterministic per asset.
HEADS = ["Vol", "Kol", "Zan", "Mir", "Lum", "Aur", "Ner", "Tas", "Bri", "Fel", "Nov", "Qua",
         "Sol", "Vex", "Pri", "Cor", "Dax", "Elo", "Fyn", "Gia", "Hal", "Ivo", "Jor", "Kes",
         "Lir", "Mox", "Nyl", "Orb", "Pel", "Riv", "Sef", "Tor", "Ulm", "Ver", "Wyn", "Xan",
         "Yur", "Zep", "Cal", "Dun", "Esk", "Fro", "Gal", "Hux", "Ind", "Jal", "Kip", "Lex"]
TAILS = ["ta", "qo", "ra", "nis", "vio", "lux", "zen", "mar", "dor", "via", "ko", "san",
         "tir", "pex", "nova", "ly", "mo", "dra", "sel", "vik", "ora", "een", "ish", "ade"]

DRINK_GROUP = {"cola", "soda", "energy", "juice", "water"}


def _rng_int(seed_text: str, salt: str, modulo: int) -> int:
    digest = hashlib.blake2b(f"{seed_text}:{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def form_of(asset: str, category: str) -> str:
    if asset in FORM_OVERRIDE:
        return FORM_OVERRIDE[asset]
    tokens = set(re.split(r"[^a-z0-9]+", asset.lower()))
    for form, keys in FORM_TOKENS:
        if tokens & keys:
            # A "can of beans" is shelved as a tin, not a beverage can.
            return "tin" if (category == "canned" and form == "can") else form
    return FORM_DEFAULT[category]


def load_english_words() -> set[str]:
    """System dictionary, used to keep invented marks from landing on real words."""
    path = Path("/usr/share/dict/american-english")
    if not path.exists():
        return set()
    return {w.strip().lower() for w in path.read_text(errors="ignore").splitlines()
            if len(w.strip()) > 3}


def wordmark(asset: str, taken: set[str], blocklist: set[str], words: set[str]) -> str:
    """Deterministic invented brand: unique, not a real mark, not a dictionary word.

    The dictionary screen matters -- an unscreened run produced FROZEN and VOLVIK,
    both of which are somebody's trademark.
    """
    for attempt in range(256):
        head = HEADS[_rng_int(asset, f"h{attempt}", len(HEADS))]
        tail = TAILS[_rng_int(asset, f"t{attempt}", len(TAILS))]
        name = (head + tail).upper()
        lowered = name.lower()
        if name in taken or lowered in words:
            continue
        if any(b in lowered for b in blocklist):
            continue
        return name
    raise RuntimeError(f"no free wordmark for {asset}")


def pick_replacement(asset: str, inv: dict, survivors: list[str], plan: dict) -> str:
    """Nearest-size survivor: same category first, then the wider drink group."""
    category, form = plan[asset]["category"], plan[asset]["form"]
    bbox = inv["assets"][asset]["bbox"]
    dims = sorted(bbox, reverse=True)
    footprint = dims[1] * dims[2]
    max_height = max(bbox) * 1.02  # 2% slack absorbs near-identical meshes

    pools = [[s for s in survivors if plan[s]["category"] == category]]
    if category in DRINK_GROUP:
        pools.append([s for s in survivors if plan[s]["category"] in DRINK_GROUP])
    pools.append(survivors)

    for pool in pools:
        scored = []
        for cand in pool:
            cbbox = inv["assets"][cand]["bbox"]
            cdims = sorted(cbbox, reverse=True)
            if cdims[1] * cdims[2] > 1.10 * footprint:
                continue  # wider than the shelf slot
            if max(cbbox) > max_height:
                continue  # taller than the original: would hit the board above
            size = sum((a - b) ** 2 for a, b in zip(dims, cdims)) ** 0.5
            scored.append((size + (0.0 if plan[cand]["form"] == form else 0.02), cand))
        if scored:
            return min(scored)[1]
    raise RuntimeError(f"no replacement fits {asset}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inventory", help="inventory.json from inventory.py")
    ap.add_argument("--out", required=True, help="Where to write rename_map.yaml")
    ap.add_argument("--blocklist", default=None, help="Optional brand blocklist (one per line)")
    args = ap.parse_args()

    inv = json.loads(Path(args.inventory).read_text())
    products = sorted(a for a, r in inv["assets"].items() if not r["is_fixture"])
    missing = set(products) - set(CATEGORY)
    if missing:
        raise SystemExit(f"uncategorised products (add to CATEGORY): {sorted(missing)}")

    blocklist = set()
    if args.blocklist:
        blocklist = {w.strip().lower() for w in Path(args.blocklist).read_text().splitlines()
                     if w.strip() and not w.startswith("#")}

    words = load_english_words()
    plan, taken, counters = {}, set(), {}
    for asset in products:
        category = CATEGORY[asset]
        form = form_of(asset, category)
        key = f"{category}_{form}"
        counters[key] = counters.get(key, 0) + 1
        brand = wordmark(asset, taken, blocklist, words)
        taken.add(brand)
        plan[asset] = {
            "category": category,
            "form": form,
            "new_id": f"{key}_{counters[key]:02d}",
            "brand": brand,
            "display": f"{brand} {DISPLAY[category]}",
            "instances": inv["assets"][asset]["instances"],
            "bbox_mm": [round(v * 1000, 1) for v in inv["assets"][asset]["bbox"]],
        }

    survivors = [a for a in products if a not in SHAPE_IP]
    for asset, reason in SHAPE_IP.items():
        if asset not in plan:
            continue
        plan[asset]["shape_ip"] = True
        plan[asset]["shape_ip_reason"] = reason
        plan[asset]["replacement"] = pick_replacement(asset, inv, survivors, plan)

    Path(args.out).write_text(yaml.safe_dump(
        {"products": plan}, sort_keys=True, default_flow_style=False, width=100))

    present = [a for a in SHAPE_IP if a in plan]
    dropped = sum(plan[a]["instances"] for a in present)
    print(f"{len(plan)} products planned; {len(present)} shape-IP drops "
          f"({dropped} item prims re-pointed)")
    for asset in SHAPE_IP:
        if asset in plan:
            p = plan[asset]
            print(f"  {asset}  x{p['instances']}  ->  {p['replacement']} "
                  f"({plan[p['replacement']]['new_id']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
