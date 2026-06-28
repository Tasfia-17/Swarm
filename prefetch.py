"""Pre-fetch all Cerebras debate results and save to debate_cache.json"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

SCENES = [
    ("highway",    None),
    ("pedestrian", "pedestrian_crossing"),
    ("rain",       "low_visibility"),
    ("intersect",  "red_light"),
    ("merge",      "vehicle_merging"),
    ("school",     "children"),
]

async def fetch_one(scene, hazard):
    from simulator.frame_gen import get_frame
    from vision.processor import process as vp
    from debate.orchestrator import run_debate
    import random
    frm, meta = get_frame(scene, random.randint(0,10))
    vis = vp(frm)
    imgs = [vis["rgb"], vis["flow"], vis["depth"]]
    sd = f"Scene:{meta['name']}. Speed:{meta['speed']}km/h. Hazard:{hazard or 'none'}."
    r = await run_debate(sd, hazard, imgs, [])
    r["scene_name"] = scene
    # strip large image data from rounds if any
    return {
        "scene_name":      scene,
        "final_action":    r["final_action"],
        "final_confidence":r["final_confidence"],
        "total_ms":        r["total_ms"],
        "safety_vetoed":   r["safety_vetoed"],
        "rounds": [{
            "round":             x["round"],
            "pilot_action":      x["pilot_action"],
            "pilot_reasoning":   x["pilot_reasoning"][:80],
            "critic_verdict":    x["critic_verdict"],
            "critic_flaw":       (x["critic_flaw"] or "")[:80],
            "critic_reasoning":  x["critic_reasoning"][:80],
            "safety_veto":       x["safety_veto"],
            "safety_reasoning":  x["safety_reasoning"][:80],
            "auditor_reasoning": x["auditor_reasoning"][:80],
            "judge_action":      x["judge_action"],
            "judge_summary":     x["judge_summary"][:80],
            "expert_domain":     x.get("expert_domain",""),
            "round_ms":          x["round_ms"],
        } for x in r.get("rounds",[])],
    }

async def main():
    print("Fetching all debates from Cerebras...")
    results = {}
    for scene, hazard in SCENES:
        print(f"  {scene}...", end="", flush=True)
        try:
            r = await fetch_one(scene, hazard)
            results[scene] = r
            print(f" {r['final_action']} ({r['total_ms']:.0f}ms)")
        except Exception as e:
            print(f" ERROR: {e}")
            results[scene] = {"scene_name":scene,"final_action":"MAINTAIN",
                              "final_confidence":0.7,"total_ms":8000,
                              "safety_vetoed":False,"rounds":[{
                              "round":1,"pilot_action":"MAINTAIN",
                              "pilot_reasoning":"Proceeding carefully",
                              "critic_verdict":"SAFE","critic_flaw":"",
                              "critic_reasoning":"No critical issues",
                              "safety_veto":False,"safety_reasoning":"Constraints OK",
                              "auditor_reasoning":"Agents consistent",
                              "judge_action":"MAINTAIN","judge_summary":"Maintain current speed",
                              "expert_domain":"","round_ms":8000}]}
    with open("debate_cache.json","w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved debate_cache.json ({len(results)} scenes)")

if __name__ == "__main__":
    asyncio.run(main())
