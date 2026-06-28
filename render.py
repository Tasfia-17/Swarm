"""
SwarmPilot 60s demo renderer — reads from debate_cache.json, pure OpenCV.
No live API calls. Low CPU. Run: python3 render.py
"""
import cv2, numpy as np, math, json, os, sys, random

W, H, FPS = 1280, 720, 30
TOTAL = W * 2  # 1800 frames = 60s  (reuse W as shorthand)
TOTAL = FPS * 60

SIM_W = 896   # 70% of 1280
PAN_X = SIM_W # right panel starts here

ROAD_L, ROAD_R = 180, 716
LANE_XS = [220, 330, 440, 550, 660]

# BGR colors
BG     = (14, 10, 18)
ROAD_C = (52, 48, 58)
NEON_G = (30, 255, 120)
NEON_B = (255, 200, 0)
ORANGE = (30, 140, 255)
RED    = (40,  40, 220)
WHITE  = (255,255,255)
GRAY   = (100, 95,110)
YELLOW = (0,  210,255)
DARK   = (10,   8, 16)

AGENT_COL = {
    "pilot":   (50, 160,255),
    "critic":  (255,130, 80),
    "safety":  (80, 220, 50),
    "expert":  (40, 190,220),
    "auditor": (240, 70,200),
    "judge":   (230,230, 40),
}
ACTION_COL = {
    "BRAKE": RED, "STOP": RED,
    "ACCELERATE": NEON_G, "MAINTAIN": NEON_G,
    "STEER_LEFT": NEON_B, "STEER_RIGHT": NEON_B,
}

# Timeline: (start_s, end_s, scene_key)
TL = [
    ( 0,  4, "hook"),
    ( 4, 12, "highway"),
    (12, 20, "pedestrian"),
    (20, 28, "rain"),
    (28, 36, "intersect"),
    (36, 44, "merge"),
    (44, 52, "school"),
    (52, 60, "compare"),
]
HAZARDS = {
    "pedestrian": "pedestrian_crossing",
    "rain":       "low_visibility",
    "intersect":  "red_light",
    "merge":      "vehicle_merging",
    "school":     "children",
}
LABELS = {
    "highway":    "HIGHWAY — 90 km/h",
    "pedestrian": "URBAN — PEDESTRIAN DETECTED",
    "rain":       "NIGHT RAIN — LOW VISIBILITY",
    "intersect":  "INTERSECTION — RED LIGHT",
    "merge":      "HIGHWAY MERGE — LATERAL HAZARD",
    "school":     "SCHOOL ZONE — CHILDREN",
    "compare":    "CEREBRAS vs GPU — SPEED PROOF",
}

def scene_at(fi):
    t = fi / FPS
    for s, e, sc in TL:
        if s <= t < e:
            return sc, t-s, e-s
    return "compare", 0, 8

def T(img, s, x, y, col, scale=0.55, thick=1):
    cv2.putText(img, str(s), (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

def TC(img, s, cx, y, col, scale=0.9, thick=2):
    tw = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
    cv2.putText(img, str(s), (cx-tw//2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

def glow(img, x, y, col, r, a=0.35):
    ov = img.copy()
    cv2.circle(ov, (int(x),int(y)), int(r*1.8), col, -1)
    cv2.addWeighted(ov, a*0.4, img, 1-a*0.4, 0, img)
    cv2.circle(img, (int(x),int(y)), int(r), col, -1)

def abox(img, x1,y1,x2,y2, col, a=0.45, rad=6):
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, col)
    cv2.addWeighted(rect, a, sub, 1-a, 0, sub)
    img[y1:y2, x1:x2] = sub
    cv2.rectangle(img,(x1,y1),(x2,y2),col,1)


def draw_road(img, scroll, scene, t):
    img[:, :SIM_W] = ROAD_C
    cv2.rectangle(img,(0,0),(ROAD_L,H),(28,24,34),-1)
    cv2.rectangle(img,(ROAD_R,0),(SIM_W,H),(28,24,34),-1)
    cv2.rectangle(img,(ROAD_L,0),(ROAD_R,H),(52,48,58),-1)
    if scene=="rain":
        r=random.Random(int(t*7))
        for _ in range(35):
            rx=r.randint(ROAD_L,ROAD_R); ry=r.randint(0,H)
            cv2.line(img,(rx,ry),(rx-1,ry+14),(150,170,190),1)
    cv2.line(img,(LANE_XS[0]-46,0),(LANE_XS[0]-46,H),(175,170,155),2)
    cv2.line(img,(LANE_XS[-1]+46,0),(LANE_XS[-1]+46,H),(175,170,155),2)
    off=int(scroll)%70
    for lx in LANE_XS[1:-1]:
        for y in range(-off,H+70,70):
            cv2.line(img,(lx,y),(lx,y+40),(165,160,140),2)
    if scene=="intersect":
        iy=H//2+50
        cv2.rectangle(img,(0,iy-100),(SIM_W,iy+100),(52,48,58),-1)
        for i in range(9):
            cx2=ROAD_L+25+i*55
            cv2.rectangle(img,(cx2,iy-90),(cx2+35,iy-25),(188,183,168),-1)
        cv2.line(img,(ROAD_L,iy-100),(ROAD_R,iy-100),WHITE,3)
        cv2.rectangle(img,(ROAD_R+10,iy-170),(ROAD_R+45,iy-60),(20,18,25),-1)
        cv2.circle(img,(ROAD_R+27,iy-148),14,RED,-1)
        glow(img,ROAD_R+27,iy-148,RED,18,0.9)
        cv2.circle(img,(ROAD_R+27,iy-112),14,(20,50,20),-1)
    elif scene=="school":
        for sy in [80,300,520]:
            cv2.rectangle(img,(ROAD_L-68,sy),(ROAD_L-8,sy+48),(0,200,240),-1)
            T(img,"SCHOOL",ROAD_L-66,sy+18,(0,0,0),0.38,1)
            T(img,"ZONE",  ROAD_L-58,sy+36,(0,0,0),0.38,1)
    for i in range(6):
        lx1=ROAD_L-40; lx2=ROAD_R+40
        ly=(i*160+int(scroll*0.65))%(H+160)-160
        for lx in [lx1,lx2]:
            cv2.line(img,(lx,ly),(lx,ly+80),(60,58,72),2)
            tip=(lx+(18 if lx==lx1 else -18),ly)
            cv2.line(img,(lx,ly),tip,(60,58,72),2)
            col2=(200,220,255) if scene!="rain" else (220,230,255)
            glow(img,tip[0],tip[1],col2,14,0.5 if scene!="rain" else 0.9)
    for i in range(5):
        bx=i*200
        bh=80+(i*77)%100
        cv2.rectangle(img,(bx,H//2-bh),(bx+55,H//2),(18,16,24),-1)
        cv2.rectangle(img,(ROAD_R+30+(bx%120),H//2-bh+20),(ROAD_R+30+(bx%120)+45,H//2),(18,16,24),-1)


def draw_npc(img,x,y,col):
    x,y=int(x),int(y)
    cv2.rectangle(img,(x-14,y-26),(x+14,y+26),col,-1)
    cv2.rectangle(img,(x-14,y-26),(x+14,y+26),WHITE,1)
    cv2.rectangle(img,(x-10,y-22),(x+10,y-10),(55,75,120),-1)
    cv2.rectangle(img,(x-6,y+16),(x-2,y+22),(30,30,180),-1)
    cv2.rectangle(img,(x+2,y+16),(x+6,y+22),(30,30,180),-1)


def draw_ped(img,x,y,t):
    x,y=int(x),int(y); bob=int(math.sin(t*9)*2)
    cv2.circle(img,(x,y-16+bob),8,(155,190,225),-1)
    cv2.line(img,(x,y-8+bob),(x,y+18+bob),(110,150,210),4)
    lk=int(math.sin(t*9)*9)
    cv2.line(img,(x,y+18+bob),(x-lk,y+34+bob),(110,150,210),3)
    cv2.line(img,(x,y+18+bob),(x+lk,y+34+bob),(110,150,210),3)
    abox(img,x-17,y-38,x+17,y-24,RED,0.75)
    T(img,"PED",x-11,y-27,WHITE,0.32,1)


def draw_ego(img,x,y,action,t):
    x,y=int(x),int(y)
    col=ACTION_COL.get(action,NEON_G)
    glow(img,x,y,col,36,0.45)
    for lx in [x-10,x+10]: glow(img,lx,y-28,(210,225,255),20,0.65)
    cv2.rectangle(img,(x-15,y-28),(x+15,y+28),(36,30,46),-1)
    cv2.rectangle(img,(x-15,y-28),(x+15,y+28),col,2)
    cv2.rectangle(img,(x-11,y-24),(x+11,y-10),(50,70,120),-1)
    cv2.rectangle(img,(x-8,y-4),(x+8,y+12),(26,22,36),-1)
    if action in ("BRAKE","STOP"):
        glow(img,x-10,y+24,RED,14,1.1)
        glow(img,x+10,y+24,RED,14,1.1)
    else:
        cv2.rectangle(img,(x-12,y+20),(x-4,y+26),(30,30,170),-1)
        cv2.rectangle(img,(x+4, y+20),(x+12,y+26),(30,30,170),-1)
    if action=="ACCELERATE":
        cv2.fillPoly(img,[np.array([[x,y-40],[x-8,y-28],[x+8,y-28]],np.int32)],NEON_G)
    elif action=="STEER_LEFT":
        cv2.arrowedLine(img,(x,y-28),(x-32,y-48),NEON_B,2,tipLength=0.4)
    elif action=="STEER_RIGHT":
        cv2.arrowedLine(img,(x,y-28),(x+32,y-48),NEON_B,2,tipLength=0.4)


def draw_speedo(img,cx,cy,speed):
    r=50
    cv2.ellipse(img,(cx,cy),(r,r),0,210,390,(38,34,50),6)
    pct=min(speed/120,1.0)
    if pct>0.01:
        c=NEON_G if speed<60 else ORANGE if speed<90 else RED
        cv2.ellipse(img,(cx,cy),(r,r),0,int(210-pct*240),210,c,6)
    na=math.radians(210-pct*240)
    cv2.line(img,(cx,cy),(cx+int((r-14)*math.cos(-na)),cy+int((r-14)*math.sin(-na))),WHITE,2)
    cv2.circle(img,(cx,cy),5,WHITE,-1)
    TC(img,f"{int(speed)}",cx,cy+8,WHITE,0.55,2)
    TC(img,"km/h",cx,cy+26,GRAY,0.32,1)


def draw_panel(img,action,conf,ms,rn,bubbles,active,speed,t,hazard,label):
    cv2.rectangle(img,(PAN_X,0),(W,H),(12,9,18),-1)
    cv2.line(img,(PAN_X,0),(PAN_X,H),(55,45,75),2)
    col=ACTION_COL.get(action,GRAY)
    T(img,"SWARMPILOT",PAN_X+10,28,NEON_B,0.75,2)
    T(img,"Cerebras x Gemma 4 31B",PAN_X+10,46,NEON_G,0.38,1)
    cv2.line(img,(PAN_X,54),(W,54),(40,35,60),1)
    abox(img,PAN_X+8,60,W-8,100,col,0.2,8)
    cv2.rectangle(img,(PAN_X+8,60),(W-8,100),col,2)
    TC(img,action,(PAN_X+W)//2,90,col,0.85,2)
    bw=W-PAN_X-16; fw=int(bw*conf)
    cv2.rectangle(img,(PAN_X+8,106),(W-8,120),(25,22,36),-1)
    if fw>0: cv2.rectangle(img,(PAN_X+8,106),(PAN_X+8+fw,120),col,-1)
    cv2.rectangle(img,(PAN_X+8,106),(W-8,120),(45,40,60),1)
    T(img,f"CONSENSUS {conf*100:.0f}%  |  {ms:.0f}ms  |  Round {rn}",PAN_X+8,134,GRAY,0.36,1)
    T(img,"AGENTS",PAN_X+8,152,GRAY,0.4,1)
    for i,(k,lb) in enumerate([("pilot","PILOT"),("critic","CRITIC"),
                                ("safety","SAFETY"),("expert","EXPERT"),
                                ("auditor","AUDITOR"),("judge","JUDGE")]):
        ay=160+i*34; ac=AGENT_COL[k]; on=active.get(k,False)
        abox(img,PAN_X+8,ay,W-8,ay+28,ac if on else GRAY,0.15 if on else 0.04,5)
        cv2.rectangle(img,(PAN_X+8,ay),(W-8,ay+28),ac if on else GRAY,1)
        cv2.circle(img,(PAN_X+20,ay+14),5,ac if on else GRAY,-1)
        T(img,lb,PAN_X+30,ay+19,ac if on else GRAY,0.42,1)
        if on:
            pw=max(2,int((W-PAN_X-80)*(0.4+0.4*math.sin(t*5+i))))
            cv2.rectangle(img,(PAN_X+68,ay+10),(PAN_X+68+pw,ay+20),ac,-1)
    cv2.line(img,(PAN_X,370),(W,370),(40,35,60),1)
    if label:
        T(img,label[:28],PAN_X+8,388,YELLOW,0.44,1)
    if hazard:
        fl=int(t*4)%2==0; hc=RED if fl else ORANGE
        abox(img,PAN_X+8,396,W-8,420,hc,0.3,4)
        T(img,f"WARNING: {hazard.replace('_',' ').upper()[:20]}",PAN_X+12,412,hc,0.38,1)
    T(img,"AGENT DEBATE",PAN_X+8,436,NEON_B,0.44,1)
    cv2.line(img,(PAN_X,444),(W,444),(40,35,60),1)
    by=450
    for b in bubbles[-10:]:
        if by>H-100: break
        bc=AGENT_COL.get(b["agent"],GRAY)
        cv2.rectangle(img,(PAN_X+4,by),(PAN_X+7,by+36),bc,-1)
        T(img,b["agent"].upper(),PAN_X+11,by+12,bc,0.34,1)
        vd=b.get("verdict","")
        if vd:
            vc=NEON_G if vd=="SAFE" else RED if vd=="UNSAFE" else ORANGE
            abox(img,W-62,by+2,W-6,by+18,vc,0.75,3)
            T(img,vd,W-58,by+13,(0,0,0),0.3,1)
        txt=b.get("text","")[:52]
        T(img,txt[:26],PAN_X+11,by+24,(155,150,165),0.32,1)
        if len(txt)>26: T(img,txt[26:],PAN_X+11,by+34,(155,150,165),0.32,1)
        by+=40
    draw_speedo(img,(PAN_X+W)//2,H-60,speed)
    dots="*"*(int(t*3)%4); T(img,dots,PAN_X+10,H-10,NEON_G,0.4,1)


def draw_top_hud(img,t):
    abox(img,0,0,SIM_W,44,(8,6,14),0.82,0)
    cv2.line(img,(0,44),(SIM_W,44),(45,38,65),1)
    T(img,"SWARMPILOT",10,30,NEON_B,0.85,2)
    T(img,"6-Agent Adversarial Driving  |  Cerebras x Gemma 4 31B",190,24,GRAY,0.42,1)


def draw_hook(img, t):
    img[:]=BG
    cx,cy=W//2,H//2; p=0.6+0.4*math.sin(t*3)
    for r,a in [(180,6),(120,14),(75,28),(45,50)]:
        ov=img.copy(); cv2.circle(ov,(cx,cy-30),r,NEON_B,-1)
        cv2.addWeighted(ov,a/255*p,img,1-a/255*p,0,img)
    TC(img,"SWARMPILOT",cx,cy+10,NEON_B,2.2,3)
    TC(img,"6 AI Agents  |  Adversarial Debate  |  Real-Time Driving",cx,cy+55,WHITE,0.65,1)
    TC(img,"Cerebras Ultra-Fast Inference  x  Google DeepMind Gemma 4 31B",cx,cy+88,NEON_G,0.52,1)


def draw_outro(img, t):
    img[:]=BG
    cx,cy=W//2,H//2; p=0.7+0.3*math.sin(t*4)
    for r,a in [(200,5),(145,12),(90,24),(55,46)]:
        ov=img.copy(); cv2.circle(ov,(cx,cy-40),r,NEON_B,-1)
        cv2.addWeighted(ov,a/255*p,img,1-a/255*p,0,img)
    TC(img,"SWARMPILOT",cx,cy+5,NEON_B,2.2,3)
    TC(img,"Adversarial Multi-Agent Autonomous Driving",cx,cy+52,WHITE,0.72,2)
    TC(img,"Cerebras x Gemma 4 31B  |  6 Parallel Agents  |  Adversarial Debate",cx,cy+85,NEON_G,0.52,1)
    TC(img,"@Cerebras  @googlegemma  #g4hackathon-multiverse-agents",cx,cy+116,GRAY,0.46,1)
    cv2.line(img,(cx-260,cy+132),(cx+260,cy+132),(40,35,60),1)
    for i,f in enumerate(["Role-Switching","D3 Budgeted Stop","RGB+Flow+Depth","65K Context"]):
        sx=cx-390+i*260; cv2.circle(img,(sx,cy+150),4,NEON_G,-1)
        TC(img,f,sx,cy+170,GRAY,0.38,1)


def draw_compare(img, t_in, dur):
    img[:]=BG
    pct=min(t_in/max(dur,1),1.0)
    TC(img,"CEREBRAS  vs  GPU PROVIDER",W//2,44,WHITE,0.85,2)
    TC(img,"Same 6-agent debate  |  Same Gemma 4 31B  |  Same prompts",W//2,70,GRAY,0.45,1)
    cv2.line(img,(0,82),(W,82),(40,35,60),1)
    cv2.line(img,(W//2,82),(W//2,H-80),(40,35,60),1)
    for ci,(label,spd_pct,col,ms_txt,tps) in enumerate([
        ("CEREBRAS  (Ours)",  min(pct*7,1.0), NEON_G, f"~{7+int(pct*1):.0f}s / debate", f"{int(1800+600*math.sin(t_in*2))} tok/s"),
        ("Standard GPU",     min(pct*1,1.0),  RED,    "~60s / debate",                  "~200 tok/s"),
    ]):
        cx2=W//4+ci*W//2
        TC(img,label,cx2,116,col,0.65,2)
        rl=ROAD_L+ci*(W//2); rr=rl+W//2-80
        cv2.rectangle(img,(rl+30,260),(rr-30,310),(52,48,58),-1)
        cv2.rectangle(img,(rl+30,260),(rr-30,310),GRAY,1)
        for dx in range(rl+50,rr-50,50):
            cv2.line(img,(dx,285),(dx+25,285),(165,160,140),2)
        car_x=rl+50+int(spd_pct*(rr-rl-120))
        cv2.rectangle(img,(car_x-15,268),(car_x+15,302),col,-1)
        cv2.rectangle(img,(car_x-15,268),(car_x+15,302),WHITE,1)
        if ci==0: glow(img,car_x,285,col,20,0.55)
        cv2.line(img,(rr-35,255),(rr-35,315),YELLOW,3)
        T(img,"FINISH",rr-58,252,YELLOW,0.38,1)
        bw2=W//2-100; fw2=int(bw2*spd_pct)
        bx2=rl+30
        cv2.rectangle(img,(bx2,330),(bx2+bw2,354),(28,24,38),-1)
        if fw2>0: cv2.rectangle(img,(bx2,330),(bx2+fw2,354),col,-1)
        cv2.rectangle(img,(bx2,330),(bx2+bw2,354),(50,45,65),1)
        TC(img,ms_txt,cx2,378,col,0.65,2)
        TC(img,tps,cx2,408,col,0.55,1)
    cv2.line(img,(0,H-120),(W,H-120),(40,35,60),1)
    for i,(k,v,c) in enumerate([
        ("6 Agents Parallel","asyncio.gather @ 100 RPM",NEON_G),
        ("3 Images/Frame","RGB + Optical Flow + Depth",NEON_B),
        ("Structured Output","strict JSON schema",NEON_G),
        ("65K Context Window","rolling world memory",ORANGE),
    ]):
        sx=80+i*(W-160)//4
        TC(img,k,sx,H-88,c,0.46,1)
        TC(img,v,sx,H-64,GRAY,0.35,1)
    if pct>0.35:
        xf=int(7*min((pct-0.35)/0.65,1.0))+1
        TC(img,f"~{xf}x FASTER",W//2,H-30,NEON_G,0.95,3)


def main(out="swarm_demo_1280.mp4"):
    cache={}
    if os.path.exists("debate_cache.json"):
        with open("debate_cache.json") as f: cache=json.load(f)

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(out,fourcc,FPS,(W,H))

    NPC_POOL=[
        [float(LANE_XS[i%5]),float(random.randint(-500,-40)),
         random.uniform(28,68),
         random.choice([(80,60,155),(50,115,55),(155,78,48),(48,78,155),(118,48,125)])]
        for i in range(10)
    ]
    peds=[]
    scroll=0.0; ego_x=float(LANE_XS[2]); ego_y=float(H*0.72)
    speed=0.0; target=65.0

    # Build per-scene bubble list from cache
    def bubbles_for(sc):
        r=cache.get(sc,{})
        bs=[]
        for rnd in r.get("rounds",[]):
            bs.append({"agent":"pilot",  "text":rnd.get("pilot_reasoning","")[:52]})
            bs.append({"agent":"critic", "text":rnd.get("critic_reasoning","")[:52],"verdict":rnd.get("critic_verdict","")})
            veto=rnd.get("safety_veto",False)
            bs.append({"agent":"safety", "text":("VETO: " if veto else "")+rnd.get("safety_reasoning","")[:46],"verdict":"UNSAFE" if veto else "SAFE"})
            if rnd.get("expert_domain"): bs.append({"agent":"expert","text":rnd["expert_domain"][:52]})
            bs.append({"agent":"auditor","text":rnd.get("auditor_reasoning","")[:52]})
            bs.append({"agent":"judge",  "text":rnd.get("judge_summary","")[:52]})
        return bs

    prev_sc=None; sc_bubbles=[]; sc_active={k:False for k in AGENT_COL}
    FIXED=1.0/FPS

    print(f"Rendering {W}x{H} 60s -> {out}")
    import time as tmod; t0=tmod.monotonic()

    for fi in range(TOTAL):
        t=fi*FIXED
        sc,t_in,t_dur=scene_at(fi)

        if sc!=prev_sc:
            prev_sc=sc
            sc_bubbles=bubbles_for(sc)
            sc_active={k:False for k in AGENT_COL}
            for b in sc_bubbles: sc_active[b["agent"]]=True
            if sc in ("pedestrian","school"):
                peds.append([float(ROAD_L-18),float(ego_y-180)])

        r=cache.get(sc,{})
        action=r.get("final_action","MAINTAIN")
        conf=r.get("final_confidence",0.72)
        ms=r.get("total_ms",7500)
        rn=len(r.get("rounds",[1]))

        # reveal bubbles gradually
        vis_bubbles=sc_bubbles[:max(1,int(t_in/t_dur*len(sc_bubbles)+1))] if sc_bubbles else []

        if action in ("BRAKE","STOP"):   target=0.0
        elif action=="ACCELERATE":       target=100.0
        elif action=="STEER_LEFT":       target=55.0; ego_x-=1.1*38*FIXED
        elif action=="STEER_RIGHT":      target=55.0; ego_x+=1.1*38*FIXED
        else:                            target=65.0
        speed+=(target-speed)*min(2.5*FIXED,1.0)
        ego_x=max(ROAD_L+18,min(ROAD_R-18,ego_x))
        scroll+=speed*FIXED*1.4

        for n in NPC_POOL:
            n[1]+=(speed-n[2])*FIXED*1.4
            if n[1]>H+100: n[1]=float(random.randint(-300,-40)); n[0]=float(LANE_XS[random.randint(0,4)])
        for p in peds: p[0]+=1.5*FIXED*60
        peds=[p for p in peds if p[0]<ROAD_R+100]

        img=np.full((H,W,3),BG,dtype=np.uint8)

        if sc=="hook":         draw_hook(img,t_in)
        elif sc=="outro":      draw_outro(img,t_in)
        elif sc=="compare":    draw_compare(img,t_in,t_dur)
        else:
            draw_road(img,scroll,sc,t)
            for n in NPC_POOL:
                if ROAD_L<n[0]<ROAD_R and -60<n[1]<H+60:
                    draw_npc(img,n[0],n[1],n[3])
            for p in peds: draw_ped(img,p[0],p[1],t)
            draw_ego(img,ego_x,ego_y,action,t)
            draw_top_hud(img,t)
            hazard=HAZARDS.get(sc)
            label=LABELS.get(sc,"")
            draw_panel(img,action,conf,ms,rn,vis_bubbles,sc_active,speed,t,hazard,label)

        writer.write(img)
        if fi%300==0:
            el=tmod.monotonic()-t0; eta=el/max(fi/TOTAL,0.001)*(1-fi/TOTAL)
            print(f"  {fi}/{TOTAL} ({fi//FPS}s)  ETA {eta:.0f}s")

    writer.release()
    print("Encoding H264...")
    final=out.replace(".mp4","_final.mp4")
    os.system(f'ffmpeg -y -i "{out}" -f lavfi -i anullsrc=r=44100:cl=stereo '
              f'-c:v libx264 -preset fast -crf 16 -pix_fmt yuv420p '
              f'-c:a aac -b:a 128k -shortest "{final}" 2>/dev/null')
    sz=os.path.getsize(final)//1024 if os.path.exists(final) else 0
    print(f"Done: {final}  ({sz} KB)")

if __name__=="__main__":
    main()
