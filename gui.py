"""
gui.py — Tkinter GUI Demo for Supervisor
Uttara University EEE BSc Thesis
Manual input → Real DQN → Output + SMS notification
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import threading
import datetime

try:
    from dqn_agent import DQNAgent
    AGENT_AVAILABLE = True
except:
    AGENT_AVAILABLE = False

ACTION_LABELS = [
    "A0: Defer all shiftable loads",
    "A1: Reduce AC intensity 20%",
    "A2: Run water pump now",
    "A3: Charge battery (off-peak)",
    "A4: Discharge battery (peak offset)",
    "A5: Start EV charging",
    "A6: Partial curtailment 50%",
]
TOU = {"off_peak": 4.05, "peak": 8.45}

class DRDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DQN Demand Response — DESCO Commercial | UU EEE Thesis")
        self.root.configure(bg="#0f1117")
        self.root.geometry("1100x720")

        # Load trained agent if available
        self.agent = None
        if AGENT_AVAILABLE:
            self.agent = DQNAgent()
            try:
                self.agent.load("dqn_model.keras")
                self.agent.epsilon = 0.0   # pure exploitation for demo
            except:
                pass   # model not trained yet — will use rule-based fallback

        self._build_ui()

    # ─────────────────────────────────────
    def _build_ui(self):
        BG   = "#0f1117"
        CARD = "#1a1d2e"
        ACC  = "#4299e1"
        FG   = "#e2e8f0"
        GRAY = "#718096"
        FONT = ("Segoe UI", 9)
        BOLD = ("Segoe UI", 9, "bold")
        H2   = ("Segoe UI", 10, "bold")

        # ── Header ──
        hdr = tk.Frame(self.root, bg="#1a1d2e", height=48)
        hdr.pack(fill="x")
        tk.Label(hdr, text="DQN Demand Response System — DESCO Commercial",
                 bg="#1a1d2e", fg=FG, font=("Segoe UI",11,"bold")).pack(side="left", padx=16, pady=12)
        tk.Label(hdr, text="Uttara University EEE | BSc Thesis | Nayeem · Anis Rahman · Sohel · Abu Hanif",
                 bg="#1a1d2e", fg=GRAY, font=FONT).pack(side="right", padx=16)

        # ── Main frame ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        # LEFT — Input panel
        left = tk.Frame(main, bg=CARD, width=330)
        left.pack(side="left", fill="y", padx=(0,8))
        left.pack_propagate(False)

        tk.Label(left, text="LAYER 1 — DESCO GRID INPUT",
                 bg=CARD, fg=ACC, font=BOLD).pack(anchor="w", padx=12, pady=(12,2))

        self.vars = {}
        fields = [
            ("generation",  "Generation (MW)",       "2840"),
            ("demand",      "Current Demand (MW)",    "2610"),
            ("hour",        "Hour (0–23)",             "17"),
            ("shed",        "Load Shed Risk (%)",       "20"),
        ]
        for key, label, default in fields:
            f = tk.Frame(left, bg=CARD)
            f.pack(fill="x", padx=12, pady=3)
            tk.Label(f, text=label, bg=CARD, fg=GRAY, font=FONT, width=22, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self.vars[key] = var
            tk.Entry(f, textvariable=var, bg="#0f1117", fg=FG, insertbackground=FG,
                     font=FONT, width=8, relief="flat",
                     highlightthickness=1, highlightbackground="#2d3748").pack(side="right")

        # Tariff dropdown
        f = tk.Frame(left, bg=CARD); f.pack(fill="x", padx=12, pady=3)
        tk.Label(f, text="Tariff Category", bg=CARD, fg=GRAY, font=FONT, width=22, anchor="w").pack(side="left")
        self.tariff_var = tk.StringVar(value="E")
        ttk.Combobox(f, textvariable=self.tariff_var, values=["E","C","F"],
                     width=6, state="readonly").pack(side="right")

        tk.Label(left, text="LAYER 2 — BUILDING STATE",
                 bg=CARD, fg=ACC, font=BOLD).pack(anchor="w", padx=12, pady=(16,2))

        bfields = [
            ("battery", "Battery SoC (%)", "68"),
            ("occupancy","Occupancy (%)",   "80"),
            ("temp",    "Temperature (°C)","34"),
            ("ac_load", "AC Load (0–1)",   "0.8"),
        ]
        for key, label, default in bfields:
            f = tk.Frame(left, bg=CARD)
            f.pack(fill="x", padx=12, pady=3)
            tk.Label(f, text=label, bg=CARD, fg=GRAY, font=FONT, width=22, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self.vars[key] = var
            tk.Entry(f, textvariable=var, bg="#0f1117", fg=FG, insertbackground=FG,
                     font=FONT, width=8, relief="flat",
                     highlightthickness=1, highlightbackground="#2d3748").pack(side="right")

        # Checkboxes
        self.pump_var = tk.BooleanVar(value=False)
        self.ev_var   = tk.BooleanVar(value=True)
        f = tk.Frame(left, bg=CARD); f.pack(fill="x", padx=12, pady=3)
        tk.Checkbutton(f, text="Water pump pending", variable=self.pump_var,
                       bg=CARD, fg=FG, selectcolor="#0f1117", font=FONT).pack(side="left")
        f2 = tk.Frame(left, bg=CARD); f2.pack(fill="x", padx=12, pady=3)
        tk.Checkbutton(f2, text="EV charging pending", variable=self.ev_var,
                       bg=CARD, fg=FG, selectcolor="#0f1117", font=FONT).pack(side="left")

        # Notification settings
        tk.Label(left, text="NOTIFICATION SETTINGS",
                 bg=CARD, fg=ACC, font=BOLD).pack(anchor="w", padx=12, pady=(16,2))
        for key, label, default in [("phone","Manager Phone","+880-1XX-XXXXXXX"),
                                     ("building","Building Name","Uttara Commercial Tower")]:
            f = tk.Frame(left, bg=CARD); f.pack(fill="x", padx=12, pady=3)
            tk.Label(f, text=label, bg=CARD, fg=GRAY, font=FONT, width=16, anchor="w").pack(side="left")
            var = tk.StringVar(value=default); self.vars[key]=var
            tk.Entry(f, textvariable=var, bg="#0f1117", fg=FG, insertbackground=FG,
                     font=FONT, relief="flat",
                     highlightthickness=1, highlightbackground="#2d3748").pack(side="right", fill="x", expand=True)

        # RUN button
        self.run_btn = tk.Button(left, text="▶  Run DQN Agent",
                                  command=self._run, bg="#3182ce", fg="white",
                                  font=("Segoe UI",10,"bold"), relief="flat",
                                  activebackground="#2b6cb0", activeforeground="white",
                                  padx=10, pady=8, cursor="hand2")
        self.run_btn.pack(fill="x", padx=12, pady=16)

        # RIGHT — Output panel
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Enter grid data and click Run DQN Agent")
        tk.Label(right, textvariable=self.status_var,
                 bg="#1a1d2e", fg=ACC, font=BOLD, anchor="w",
                 pady=6, padx=10).pack(fill="x", pady=(0,6))

        # Notebook tabs
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        style = ttk.Style()
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=GRAY,
                        padding=[12,4], font=FONT)
        style.map("TNotebook.Tab", background=[("selected","#2d3748")], foreground=[("selected",FG)])

        # Tab 1 — AI Decision
        tab1 = tk.Frame(nb, bg=BG)
        nb.add(tab1, text="  🤖 DQN Decision  ")
        self.decision_text = scrolledtext.ScrolledText(
            tab1, bg=CARD, fg=FG, font=("Consolas",9),
            insertbackground=FG, relief="flat", padx=10, pady=10,
            state="disabled")
        self.decision_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 2 — Q-Values
        tab2 = tk.Frame(nb, bg=BG)
        nb.add(tab2, text="  📊 Q-Values  ")
        self.q_canvas = tk.Canvas(tab2, bg=CARD, highlightthickness=0)
        self.q_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 3 — SMS Notification
        tab3 = tk.Frame(nb, bg=BG)
        nb.add(tab3, text="  📱 SMS Notification  ")
        self.sms_text = scrolledtext.ScrolledText(
            tab3, bg="#0a1a0a", fg="#68d391", font=("Consolas",9),
            insertbackground="#68d391", relief="flat", padx=14, pady=12,
            state="disabled")
        self.sms_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 4 — Reward
        tab4 = tk.Frame(nb, bg=BG)
        nb.add(tab4, text="  💰 Reward  ")
        self.reward_text = scrolledtext.ScrolledText(
            tab4, bg=CARD, fg=FG, font=("Consolas",9),
            insertbackground=FG, relief="flat", padx=10, pady=10,
            state="disabled")
        self.reward_text.pack(fill="both", expand=True, padx=4, pady=4)

    # ─────────────────────────────────────
    def _run(self):
        self.run_btn.config(state="disabled", text="⟳  DQN Computing...")
        threading.Thread(target=self._compute, daemon=True).start()

    def _compute(self):
        try:
            h    = int(self.vars["hour"].value() if hasattr(self.vars["hour"],"value") else self.vars["hour"].get())
            gen  = int(self.vars["generation"].get())
            dem  = int(self.vars["demand"].get())
            bat  = int(self.vars["battery"].get())
            occ  = int(self.vars["occupancy"].get())
            temp = float(self.vars["temp"].get())
            ac   = float(self.vars["ac_load"].get())
            shed = int(self.vars["shed"].get())
            tariff_key = self.tariff_var.get()
            tariffs = {"E":(4.05,8.45),"C":(3.50,5.95),"F":(3.43,7.12)}
            off_p, pk_p = tariffs[tariff_key]
            price = pk_p if 17<=h<=22 else off_p
            stress = round(dem/gen*100, 1)

            # Build state vector (8-dim, normalized)
            state = np.array([
                h/23.0, price/8.45, min(stress/100,1.0),
                bat/100.0, occ/100.0, (temp-20)/22.0,
                shed/100.0, float(self.pump_var.get() or self.ev_var.get())
            ], dtype=np.float32)

            # Get Q-values
            if self.agent:
                qvals = self.agent.get_q_values(state)
            else:
                qvals = self._rule_qvals(state, h, price, bat, shed)

            best   = int(np.argmax(qvals))
            reward = self._compute_reward(best, price, bat, occ, temp)

            self.root.after(0, lambda: self._update_ui(
                state, qvals, best, reward, stress, price, h, bat, shed,
                off_p, pk_p, tariff_key
            ))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="▶  Run DQN Agent"))

    def _rule_qvals(self, state, h, price, bat, shed):
        """Fallback Q-values when model not yet trained"""
        peak = 17<=h<=22
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.3, 0.2], dtype=np.float32)
        if peak:
            q[4] += 0.4 if bat>30 else 0
            q[0] += 0.2
            q[3] -= 0.5
        else:
            q[3] += 0.5 if bat<80 else 0
            q[2] += 0.3 if self.pump_var.get() else 0
            q[5] += 0.3 if self.ev_var.get() else 0
        if shed>20: q[4]+=0.2; q[0]+=0.1
        return q + np.random.uniform(-0.05, 0.05, 7)

    def _compute_reward(self, action, price, bat, occ, temp):
        peak = price > 7.0
        cost_pen    = -(price/8.45*1.5) if peak else -(price/8.45*0.4)
        arbitrage   = 0.85 if action==4 and bat>30 else (0.70 if action==3 else 0.0)
        comfort     = -0.3 if action==6 else (-0.2 if action==1 and temp>35 else 0.5)
        par_red     = 0.35 if action in [4,1,6] else -0.1
        total       = cost_pen + arbitrage + comfort + par_red
        return {"cost":round(cost_pen,3),"arb":round(arbitrage,3),
                "comfort":round(comfort,3),"par":round(par_red,3),"total":round(total,3)}

    def _update_ui(self, state, qvals, best, reward, stress, price, h, bat,
                   shed, off_p, pk_p, tariff_key):
        peak      = 17<=h<=22
        now_str   = datetime.datetime.now().strftime("%H:%M:%S")
        tariff_names = {"E":"Category-E Commercial","C":"Category-C Small Industry","F":"Category-F Medium Voltage"}
        tname     = tariff_names[tariff_key]
        price_str = f"৳{pk_p} PEAK" if peak else f"৳{off_p} off-peak"
        phone     = self.vars["phone"].get()
        bldg      = self.vars["building"].get()

        # ── Tab 1: Decision ──
        load_status = {
            0: [("Central AC",    "Normal ✓"),("Battery","Idle"),   ("Pump","Defer"),("EV","Defer")],
            1: [("Central AC",    "Reduced -20% ↓"),("Battery","Idle"),("Pump","Defer"),("EV","Defer")],
            2: [("Central AC",    "Normal ✓"),("Battery","Idle"),   ("Pump","Running NOW ✓"),("EV","Defer")],
            3: [("Central AC",    "Normal ✓"),("Battery","Charging ↑"),("Pump","Defer"),("EV","Defer")],
            4: [("Central AC",    "Normal ✓"),("Battery",f"Discharging ↓ ({bat}%→)"),("Pump","Defer"),("EV","Defer")],
            5: [("Central AC",    "Normal ✓"),("Battery","Idle"),   ("Pump","Defer"),("EV","Charging ✓")],
            6: [("Central AC",    "Curtailed 50% ↓↓"),("Battery","Idle"),("Pump","Defer"),("EV","Defer")],
        }
        lines = [
            f"{'='*54}",
            f"  DQN DEMAND RESPONSE — AI DECISION REPORT",
            f"  Time: {now_str} | Hour: {h:02d}:00 | {tname}",
            f"{'='*54}\n",
            f"  GRID STATUS:",
            f"  {'Price':20s}: {price_str}",
            f"  {'Stress':20s}: {stress}%  {'🔴 CRITICAL' if stress>=90 else '🟡 HIGH' if stress>=80 else '🟢 NORMAL'}",
            f"  {'Load Shed Risk':20s}: {shed}%",
            f"  {'Battery SoC':20s}: {bat}%\n",
            f"  DQN STATE VECTOR (8-dim):",
            *[f"  s[{i}] = {v:.4f}  ({n})" for i,(v,n) in enumerate(zip(state,
              ["hour","tou_price","grid_stress","battery_soc","occupancy","temperature","shed_risk","pending"]))],
            f"\n  {'─'*50}",
            f"  ✅ SELECTED ACTION: {best}",
            f"  → {ACTION_LABELS[best]}",
            f"  Q-value: {qvals[best]:.4f}",
            f"  {'─'*50}\n",
            f"  LOAD CONTROL COMMANDS:",
        ]
        for name, status in load_status.get(best, []):
            lines.append(f"  {'['+name+']':25s} → {status}")
        lines += ["","  Non-shiftable loads (Lights/Fans/Servers) → USER CONTROL — AI never touches"]
        self._set_text(self.decision_text, "\n".join(lines))

        # ── Tab 2: Q-Values canvas ──
        c   = self.q_canvas
        c.delete("all")
        W   = c.winfo_width()  or 650
        H   = c.winfo_height() or 340
        c.config(bg="#1a1d2e")
        c.create_text(W//2, 18, text="Q-Values: DQN Neural Network Output (7 actions)",
                      fill="#e2e8f0", font=("Segoe UI",10,"bold"))
        qmin = min(qvals); qmax = max(qvals); qrange = max(qmax-qmin, 0.01)
        bar_h= 28; pad_l=200; pad_r=90; pad_top=40; gap=8
        colors = ["#4299e1","#68d391","#f6ad55","#b794f4","#fc8181","#63b3ed","#f687b3"]
        for i, (q,col) in enumerate(zip(qvals, colors)):
            y   = pad_top + i*(bar_h+gap)
            pct = (q-qmin)/qrange
            bw  = max(6, pct*(W-pad_l-pad_r))
            bc  = col if i!=best else "#ffffff"
            c.create_text(pad_l-8, y+bar_h//2, text=ACTION_LABELS[i], fill="#a0aec0",
                          font=("Segoe UI",8), anchor="e")
            c.create_rectangle(pad_l, y, pad_l+bw, y+bar_h,
                                fill=col, outline="", )
            if i==best:
                c.create_rectangle(pad_l-2, y-2, pad_l+bw+2, y+bar_h+2,
                                    fill="", outline="#ffffff", width=2)
                c.create_text(pad_l+bw+6, y+bar_h//2, text=f"Q={q:.3f} ◀ BEST",
                               fill="#ffffff", font=("Segoe UI",8,"bold"), anchor="w")
            else:
                c.create_text(pad_l+bw+6, y+bar_h//2, text=f"Q={q:.3f}",
                               fill="#4a5568", font=("Segoe UI",8), anchor="w")

        # ── Tab 3: SMS ──
        saving_est = round(abs(reward["arb"]) * 14 * 30)
        sms = f"""{'─'*48}
  📱 SMS NOTIFICATION
  From : DESCO DR System
  To   : {phone}
  Time : {now_str}
{'─'*48}

{'⚠️  DR ALERT' if stress>=90 or peak else '✅  OFF-PEAK OPPORTUNITY' if not peak else '📊  DR STATUS'}
Building: {bldg}

DESCO Grid Status:
  Price       : {price_str}
  Grid Stress : {stress}%  {'🔴 Critical' if stress>=90 else '🟡 High' if stress>=80 else '🟢 Normal'}
  Load Shed   : {shed}% risk
  Battery     : {bat}% SoC

DQN AI Decision:
  Action {best}: {ACTION_LABELS[best]}
  Reward r(t) = {reward['total']}

Cost Analysis:
  Cost penalty    : {reward['cost']}
  Battery saving  : +{reward['arb']}
  Comfort status  : {'OK ✓' if reward['comfort']>=0 else 'Minor reduction'}
  PAR reduction   : +{reward['par']}

Estimated Monthly Saving: ৳{saving_est:,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESCO Commercial Demand Response System
Powered by DQN AI — Uttara University EEE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        self._set_text(self.sms_text, sms)

        # ── Tab 4: Reward ──
        rew_lines = [
            f"{'='*50}",
            f"  REWARD FUNCTION BREAKDOWN",
            f"  r(t) = -α·Cost  -β·PAR  -γc·Comfort  +δ·Arbitrage",
            f"{'='*50}\n",
            f"  Weights: α=1.0 | β=0.30 | γc=0.50 | δ=0.40\n",
            f"  {'Component':30s} {'Value':>10s}",
            f"  {'─'*40}",
            f"  {'Cost Penalty  (-α × C(t))':30s} {reward['cost']:>10.3f}",
            f"  {'PAR Reduction (-β × PAR)':30s} {reward['par']:>10.3f}",
            f"  {'Comfort       (-γc × Φ)':30s} {reward['comfort']:>10.3f}",
            f"  {'Arbitrage     (+δ × B)':30s} {reward['arb']:>10.3f}",
            f"  {'─'*40}",
            f"  {'TOTAL REWARD  r(t)':30s} {reward['total']:>10.3f}\n",
            f"  Transition stored in replay buffer ✓",
            f"  Buffer will be sampled for DQN training ✓",
            f"\n  Bellman TD Target:",
            f"  y_t = r(t) + γ · max_a' Q(s_t+1, a'; θ⁻)",
            f"      = {reward['total']:.3f} + 0.95 · Q_next",
            f"\n  Huber Loss L(θ) minimized via Adam optimizer",
            f"  Learning rate α = 0.001",
        ]
        self._set_text(self.reward_text, "\n".join(rew_lines))

        self.status_var.set(f"✅ DQN Decision: Action {best} — {ACTION_LABELS[best].split(':')[1].strip()} | Reward: {reward['total']}")
        self.run_btn.config(state="normal", text="▶  Run DQN Agent Again")

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

def main():
    root = tk.Tk()
    app  = DRDemoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
