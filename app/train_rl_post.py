import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import os


class RLEnvironment:
    """
    Environment for RL-based post-training of LLM
    Rewards model for using preferred response formats
    """
    def __init__(self, max_seq_length=50):
        self.max_seq_length = max_seq_length
        self.state = []
        
        # Define preferred response patterns
        self.good_starts = [
            "That is a great question",
            "Let me think about that"
        ]
        self.good_ends = [
            "let me know if you have any other questions"
        ]
    
    def reset(self, initial_tokens):
        """Reset environment with initial prompt tokens"""
        self.state = list(initial_tokens)
        return self.state
    
    def step(self, action_token):
        """Take a step by adding a token to the sequence"""
        self.state.append(action_token)
        
        # Check if sequence is complete
        done = len(self.state) >= self.max_seq_length
        
        # Compute reward if done
        reward = 0.0
        if done:
            reward = self.compute_reward(self.state)
        
        return self.state, reward, done
    
    def compute_reward(self, token_sequence):
        """
        Compute reward based on response format
        Higher reward for responses with preferred format
        """
        # Decode tokens to text
        text = self.tokenizer.decode(token_sequence, skip_special_tokens=True).lower()
        
        reward = 0.0
        
        # Reward for good start phrases
        for start_phrase in self.good_starts:
            if text.startswith(start_phrase.lower()):
                reward += 10.0
                break
        
        # Reward for good end phrases
        for end_phrase in self.good_ends:
            if text.endswith(end_phrase.lower()):
                reward += 10.0
                break
        
        # Penalty for too short responses
        if len(token_sequence) < 10:
            reward -= 5.0
        
        # Penalty for too repetitive text
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                reward -= 5.0
        
        return reward
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for decoding"""
        self.tokenizer = tokenizer


class RLPolicy:
    """
    Policy for RL-based text generation
    Uses the base LLM and adapts it with RL
    """
    def __init__(self, model, tokenizer, optimizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
    
    def get_action(self, token_sequence, temperature=1.0):
        """
        Generate next token using the model
        Returns: (token_id, log_probability)
        """
        # Convert to tensor
        input_ids = torch.tensor([token_sequence], dtype=torch.long).to(self.device)
        
        # Get model output
        outputs = self.model(input_ids)
        logits = outputs.logits[0, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample token
        dist = torch.distributions.Categorical(probs)
        token_id = dist.sample()
        log_prob = dist.log_prob(token_id)
        
        return token_id.item(), log_prob
    
    def compute_loss(self, log_probs, rewards):
        """Compute policy gradient loss"""
        # Normalize rewards
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if rewards_tensor.std() > 0:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Compute loss
        loss = -(torch.stack(log_probs) * rewards_tensor).mean()
        return loss
    
    def train_one_epoch(self, env, batch_size, questions):
        """
        Train for one epoch using policy gradient
        """
        batch_log_probs = []
        batch_rewards = []
        batch_lens = []
        
        for _ in range(batch_size):
            # Sample a random question
            question = random.choice(questions)
            
            # Encode question
            input_ids = self.tokenizer.encode(question, return_tensors='pt')[0].tolist()
            
            # Reset environment
            state = env.reset(input_ids)
            done = False
            episode_log_probs = []
            
            # Generate response
            while not done and len(state) < env.max_seq_length:
                token_id, log_prob = self.get_action(state, temperature=1.0)
                state, reward, done = env.step(token_id)
                episode_log_probs.append(log_prob)
            
            # Store episode data
            batch_log_probs.extend(episode_log_probs)
            batch_rewards.extend([reward] * len(episode_log_probs))
            batch_lens.append(len(episode_log_probs))
        
        # Update policy
        if len(batch_log_probs) > 0:
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch_log_probs, batch_rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            avg_len = sum(batch_lens) / len(batch_lens) if batch_lens else 0
            
            return loss.item(), avg_reward, avg_len
        
        return 0.0, 0.0, 0.0


def train_rl_post_training(model_path, output_path, epochs=50, batch_size=10, device='cuda'):
    """
    Main function to perform RL post-training
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load fine-tuned checkpoint if exists
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup environment and policy
    env = RLEnvironment(max_seq_length=50)
    env.set_tokenizer(tokenizer)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    policy = RLPolicy(model, tokenizer, optimizer, device)
    
    # Sample questions for training
    sample_questions = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "What is the capital of France?",
        "Explain quantum computing.",
        "What causes climate change?",
        "How do vaccines work?",
        "What is artificial intelligence?",
        "Describe the water cycle.",
        "What is DNA?",
        "How does the internet work?"
    ]
    
    print(f"\nStarting RL post-training for {epochs} epochs...")
    print(f"Device: {device}")
    
    # Training loop
    for epoch in range(epochs):
        loss, avg_reward, avg_len = policy.train_one_epoch(env, batch_size, sample_questions)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_len:.1f}")
    
    # Save the RL-trained model
    print(f"\nSaving RL-trained model to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_path)
    
    print("RL post-training complete!")
    
    # Test generation
    print("\nTesting generation with RL-trained model:")
    test_question = "What is reinforcement learning?"
    input_ids = tokenizer.encode(test_question, return_tensors='pt').to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nQuestion: {test_question}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    # Configuration
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    # Paths
    fine_tuned_model_path = "app/checkpoints/gpt2-squad/checkpoint_epoch_4.pth"
    rl_output_path = "app/checkpoints/gpt2-rl/rl_model.pth"
    
    # Run RL post-training
    train_rl_post_training(
        model_path=fine_tuned_model_path,
        output_path=rl_output_path,
        epochs=50,
        batch_size=10,
        device=device
    )
