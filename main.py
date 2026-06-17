"""
main.py — Entry Point
Run this file to start everything
"""
import sys

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║   DQN Demand Response — DESCO Commercial             ║
║   Uttara University EEE | BSc Thesis                 ║
║   Authors: Nayeem, Anis Rahman, Sohel, Abu Hanif     ║
║   Supervisor: Md Siam Ahmed                          ║
╚══════════════════════════════════════════════════════╝

Choose what to do:
  1 → Train DQN agent (500 episodes — quick test)
  2 → Train DQN agent (2500 episodes — full thesis)
  3 → Open GUI demo (requires trained model or uses fallback)
  4 → Run all baselines comparison
  q → Quit
""")
    choice = input("  Enter choice: ").strip()

    if choice == "1":
        from train import train
        import train as tr
        tr.EPISODES = 500
        train()

    elif choice == "2":
        from train import train
        import train as tr
        tr.EPISODES = 2500
        train()

    elif choice == "3":
        from gui import main as gui_main
        gui_main()

    elif choice == "4":
        from train import run_baseline_uncontrolled, run_baseline_rule
        from dqn_agent import DQNAgent
        from environment import DescoCommercialEnv
        import numpy as np

        print("\n  Running baselines (100 episodes each)...")
        c_unc  = run_baseline_uncontrolled(100)
        c_rule = run_baseline_rule(100)
        print(f"\n  Uncontrolled : ৳{c_unc:.2f}/day")
        print(f"  Rule-Based   : ৳{c_rule:.2f}/day")
        print(f"\n  (Train the DQN first to see DQN results)")

    elif choice == "q":
        sys.exit(0)
    else:
        print("  Invalid choice.")

if __name__ == "__main__":
    main()
