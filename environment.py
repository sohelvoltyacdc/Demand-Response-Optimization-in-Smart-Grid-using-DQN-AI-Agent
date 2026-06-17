"""
environment.py — DESCO Commercial DR Environment
Uttara University EEE BSc Thesis
Authors: Nayeem, Anis Rahman, Sohel Hossain, Abu Hanif
Supervisor: Md Siam Ahmed
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

TOU_PRICES = {"off_peak": 4.05, "peak": 8.45}
SHED_PROB  = {**{h:0.05 for h in range(0,7)}, **{h:0.08 for h in range(7,17)},
               **{h:0.20 for h in range(17,23)}, 23:0.05}
BATTERY_CAP = 5.0

class DescoCommercialEnv(gym.Env):
    """
    State (8-dim, normalized 0-1):
      [hour, tou_price, grid_stress, battery_soc,
       occupancy, temperature, shed_risk, pending_tasks]
    Actions (7):
      0=Defer all  1=Reduce AC  2=Run pump  3=Charge bat
      4=Discharge bat  5=EV charge  6=Partial curtail
    Reward:
      r = -alpha*Cost - beta*PAR - gamma*Comfort + delta*Arbitrage
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)
        # Reward weights
        self.alpha=1.0; self.beta=0.30; self.gamma_c=0.50; self.delta=0.40
        self.reset()

    def _price(self, h): return TOU_PRICES["peak"] if 17<=h<=22 else TOU_PRICES["off_peak"]
    def _stress(self):
        if 17<=self.hour<=22: return float(np.clip(0.90+np.random.uniform(-0.03,0.05),0,1))
        if  7<=self.hour<=16: return float(np.clip(0.80+np.random.uniform(-0.05,0.05),0,1))
        return float(np.clip(0.65+np.random.uniform(-0.05,0.05),0,1))

    def _obs(self):
        p=self._price(self.hour); pending=float(self.pump_pending or self.ev_pending)
        return np.clip(np.array([
            self.hour/23.0, p/8.45, self.grid_stress,
            self.battery_soc/100.0, self.occupancy/100.0,
            (self.temperature-20)/22.0, SHED_PROB.get(self.hour,0.08), pending
        ], dtype=np.float32), 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep=0; self.hour=0
        self.battery_soc=float(np.random.randint(30,70))
        self.occupancy=10.0; self.temperature=float(np.random.uniform(28,36))
        self.grid_stress=self._stress()
        self.pump_pending=True; self.ev_pending=True
        self.total_cost=0.0; self.total_comfort_vio=0
        self.load_history=[]; self.reward_history=[]
        return self._obs(), {}

    def step(self, action):
        price=self._price(self.hour); is_peak=17<=self.hour<=22
        shed=np.random.binomial(1, SHED_PROB.get(self.hour,0.05))
        occ=self.occupancy/100.0
        base_load=0.40*occ+0.60  # lights + servers
        ac_load=1.50*occ
        shift_load=0.0; comfort_vio=0.0; arbitrage_val=0.0

        if shed==1:
            self.load_history.append(0.0); self._advance()
            return self._obs(), -2.0, self.timestep>=96, False, {"hour":self.hour,"cost":self.total_cost,"shed":True}

        if action==0:
            total_load=base_load+ac_load
            if self.hour>=22 and self.pump_pending: comfort_vio+=1.0; self.total_comfort_vio+=1
            if self.hour>=22 and self.ev_pending:   comfort_vio+=1.0; self.total_comfort_vio+=1
        elif action==1:
            total_load=base_load+ac_load*0.80
            if self.temperature>35: comfort_vio+=0.2
        elif action==2:
            if self.pump_pending: shift_load=0.55; self.pump_pending=False
            total_load=base_load+ac_load+shift_load
        elif action==3:
            if self.battery_soc<90: self.battery_soc=min(100.0,self.battery_soc+10); shift_load=1.0
            total_load=base_load+ac_load+shift_load
            if is_peak: comfort_vio+=0.3
        elif action==4:
            if self.battery_soc>10:
                discharge=min(0.80,self.battery_soc/10)
                self.battery_soc=max(0.0,self.battery_soc-discharge*10)
                arbitrage_val=discharge*(price-TOU_PRICES["off_peak"])
                shift_load=-discharge
            total_load=max(0.0,base_load+ac_load+shift_load)
        elif action==5:
            if self.ev_pending:
                shift_load=1.20; self.ev_pending=False
                if is_peak: comfort_vio+=0.2
            total_load=base_load+ac_load+shift_load
        elif action==6:
            total_load=(base_load+ac_load)*0.50; comfort_vio+=0.30
        else:
            total_load=base_load+ac_load

        cost=price*total_load*0.25; self.total_cost+=cost
        self.load_history.append(total_load)
        avg_load=np.mean(self.load_history) if self.load_history else total_load
        par_pen=max(0.0,(total_load-avg_load)/(avg_load+1e-6))
        reward=(-self.alpha*cost - self.beta*par_pen - self.gamma_c*comfort_vio + self.delta*arbitrage_val)
        self.reward_history.append(reward)
        self._advance()
        done=self.timestep>=96
        return self._obs(), reward, done, False, {"hour":self.hour,"cost":self.total_cost,"load":total_load,"battery_soc":self.battery_soc,"shed":False}

    def _advance(self):
        self.timestep+=1; self.hour=(self.timestep*15)//60
        occ_map={0:5,1:5,2:5,3:5,4:5,5:5,6:15,7:40,8:75,9:90,10:95,11:95,12:70,
                 13:80,14:90,15:85,16:70,17:50,18:30,19:20,20:15,21:10,22:5,23:5}
        self.occupancy=float(occ_map.get(self.hour,10))
        self.grid_stress=self._stress()
        self.temperature=float(np.clip(self.temperature+np.random.uniform(-0.3,0.3),20,42))
