print("Starting Kevin...")

import yaml
from modules.dl_nn_strategy_generator import DLNNStrategyGenerator
from modules.dl_nn_code_generator import DLNNCodeGenerator
from modules.dl_nn_backtester import DLNNBacktester
from modules.dl_nn_optimizer import DLNNOptimizer
from modules.dl_nn_deployer import DLNNDeployer
from modules.dl_nn_trainer import DLNNTrainer
import modules.trading_env

def run_kevin_once(config, run_num):
    print(f"\nRun {run_num + 1}/3, run_num passed: {run_num}")
    print("Training models...")
    trainer = DLNNTrainer(config, run_num)
    trainer.train_models()

    print("Generating strategy...")
    sg = DLNNStrategyGenerator(config, run_num)
    strategy = sg.generate_strategy()
    print(f"Strategy: {strategy}")

    print("Generating bot code...")
    cg = DLNNCodeGenerator(config)
    bot_path = cg.generate_bot_code(strategy)
    print(f"Bot path: {bot_path}")

    print("Running backtest...")
    bt = DLNNBacktester(config, run_num)
    results = bt.run_backtest(bot_path)
    print(f"Backtest results: {results}")

    print("Optimizing strategy...")
    opt = DLNNOptimizer(config, run_num)
    optimized_strategy = opt.optimize_strategy(strategy, results)
    print(f"Optimized strategy: {optimized_strategy}")

    if optimized_strategy["profitable"]:
        print("Regenerating and deploying bot...")
        cg.generate_bot_code(optimized_strategy)
        dp = DLNNDeployer(config)
        dp.deploy_bot(bot_path)
    else:
        print("Strategy not profitable, skipping deployment.")

def run_kevin():
    print("Loading config...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    for i in range(3):
        run_kevin_once(config, i)

if __name__ == "__main__":
    run_kevin()
