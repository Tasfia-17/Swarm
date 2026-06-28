import os

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "csk-ppmxvkv62tehywwtv56h3r3fjcrvddjmw83emhv22cwmry65")
MODEL = "gemma-4-31b"
MAX_DEBATE_ROUNDS = 3
CONVERGENCE_THRESHOLD = 2  # consecutive agreements to stop early
