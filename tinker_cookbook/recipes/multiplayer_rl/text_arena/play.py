"""Interactive play and evaluation for trained tic-tac-toe policies.

The trained policy is sampled on Fireworks serverless training.
``checkpoint_path`` is the state path of a checkpoint from a serverless
training run (``state_path`` in the run's ``checkpoints.jsonl``); it is loaded
into a new serverless session and sampled from there.

Usage:

    # Play against the trained model (you go first as Player 0):
    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<state_path>

    # Play as Player 1 (model goes first):
    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<state_path> human_player_id=1

    # Evaluate against different opponents (20 games each):
    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<state_path> mode=eval opponent=random num_games=20

    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<state_path> mode=eval opponent=optimal num_games=20

    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<state_path> mode=eval opponent=base_model num_games=20
"""

import asyncio
from typing import Literal

import chz
from fireworks.training.sdk import FiretitanServiceClient
from termcolor import colored

from tinker_cookbook import model_info
from tinker_cookbook.completers import MessageCompleter, TinkerTokenCompleter
from tinker_cookbook.fireworks_utils import FIREWORKS_SERVERLESS_TRAINING_BASE_URL
from tinker_cookbook.recipes.multiplayer_rl.text_arena.env import (
    STOP_CONDITION,
    TwoPlayerEnvGroupBuilder,
)
from tinker_cookbook.recipes.multiplayer_rl.text_arena.tictactoe import (
    OpponentType,
    make_opponent,
)
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.play_w_env import ManualPolicy, print_trajectory_summary
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer


async def play_game(
    renderer: Renderer,
    model_token_completer: TinkerTokenCompleter,
    human_player_id: int,
    game_name: str,
) -> None:
    """Play an interactive game: human vs trained model."""
    model_player_id = 1 - human_player_id

    # self_play=True so both envs share one coordinator (same board).
    # Both sides run via do_single_rollout — human with ManualPolicy, model with TinkerTokenCompleter.
    builder = TwoPlayerEnvGroupBuilder(
        game_name=game_name,
        renderer=renderer,
        num_envs=2,
        self_play=True,
    )
    envs = await builder.make_envs()
    human_env = envs[human_player_id]
    model_env = envs[model_player_id]
    human_policy = ManualPolicy(renderer.tokenizer, multiline=False, show_observation=True)

    print(colored(f"\nYou are Player {human_player_id}.", "cyan", attrs=["bold"]))
    print(colored("Type moves like: [4]", "cyan"))
    print(colored("=" * 40, "cyan"))

    human_traj, _model_traj = await asyncio.gather(
        do_single_rollout(human_policy, human_env),
        do_single_rollout(model_token_completer, model_env),
    )

    print(colored("\n--- Your Game Summary ---", "cyan", attrs=["bold"]))
    print_trajectory_summary(human_traj)


async def eval_model(
    renderer: Renderer,
    model_token_completer: TinkerTokenCompleter,
    opponent_policy: MessageCompleter,
    opponent_name: str,
    game_name: str,
    num_games: int = 20,
) -> None:
    """Evaluate the trained model against a given opponent."""
    wins, losses, draws = 0, 0, 0

    for game_idx in range(num_games):
        print(colored(f"\n{'=' * 40} Game {game_idx + 1} {'=' * 40}", "cyan", attrs=["bold"]))

        builder = TwoPlayerEnvGroupBuilder(
            game_name=game_name,
            renderer=renderer,
            num_envs=2,
            self_play=False,
            opponent_policy=opponent_policy,
        )
        envs = await builder.make_envs()
        model_env = envs[0]  # Model plays as Player 0

        model_traj = await do_single_rollout(model_token_completer, model_env)

        total_reward = sum(t.reward for t in model_traj.transitions)
        if total_reward > 0:
            result = colored("WIN", "green")
            wins += 1
        elif total_reward < 0:
            result = colored("LOSS", "red")
            losses += 1
        else:
            result = colored("DRAW", "yellow")
            draws += 1
        print(f"  Result: {result} (reward={total_reward:.1f})")

    print(colored(f"\n{'=' * 40} Summary (vs {opponent_name}) {'=' * 40}", "cyan", attrs=["bold"]))
    print(f"  Wins: {wins}/{num_games}, Draws: {draws}/{num_games}, Losses: {losses}/{num_games}")


@chz.chz
class PlayConfig:
    # State path of the trained policy's checkpoint, from serverless training.
    checkpoint_path: str
    # Fireworks base model of the policy and of the base_model opponent.
    fireworks_base_model: str = "accounts/fireworks/models/qwen3p8-27b"
    lora_rank: int = 32
    # The policy and the base_model opponent are sampled on serverless training.
    base_url: str = FIREWORKS_SERVERLESS_TRAINING_BASE_URL
    model_name: str = "Qwen/Qwen3.8-27B"
    renderer_name: str | None = "qwen3_8_disable_thinking"
    game_name: str = "TicTacToe-v0"
    mode: Literal["human_vs_model", "eval"] = "human_vs_model"
    opponent: OpponentType = "random"
    human_player_id: int = 0  # 0 = go first, 1 = go second
    num_games: int = 20


async def main(config: PlayConfig) -> None:
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    renderer = get_renderer(renderer_name, get_tokenizer(config.model_name))

    # A session samples only its own checkpoints, so the trained weights are
    # loaded into this session and saved as a sampler snapshot here.
    service_client = FiretitanServiceClient(base_url=config.base_url)
    training_client = service_client.create_training_client(
        base_model=config.fireworks_base_model, lora_rank=config.lora_rank
    )
    training_client.load_state(config.checkpoint_path).result()
    sampler_path = training_client.save_weights_for_sampler("policy").result().path
    sampling_client = service_client.create_sampling_client(
        model_path=sampler_path, tokenizer=get_tokenizer(config.model_name)
    )
    model_token_completer = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=64,
    )

    if config.mode == "human_vs_model":
        while True:
            await play_game(
                renderer, model_token_completer, config.human_player_id, config.game_name
            )
            again = input(colored("\nPlay again? [y/n]: ", "cyan")).strip().lower()
            if again != "y":
                break
    elif config.mode == "eval":
        opponent_policy = make_opponent(
            config.opponent,
            config.model_name,
            renderer,
            config.fireworks_base_model,
            stop_condition=STOP_CONDITION,
            base_url=config.base_url,
        )
        await eval_model(
            renderer,
            model_token_completer,
            opponent_policy,
            config.opponent,
            config.game_name,
            config.num_games,
        )


if __name__ == "__main__":
    config = chz.entrypoint(PlayConfig)
    asyncio.run(main(config))
