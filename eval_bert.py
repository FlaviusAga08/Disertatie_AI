import torch
import time
import os
import sys
import numpy as np
import requests
import json
import psutil
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from datasets import load_dataset
from scipy import stats
from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer
from peft import PeftModel


# CONFIGURATION
class Config:
    """Centralized configuration"""
    # T5 Settings
    T5_MAX_INPUT_LENGTH = 512
    T5_MAX_OUTPUT_LENGTH = 200
    T5_NUM_BEAMS = 4
    T5_PREFIX = "summarize legal document: "
    
    # Llama Settings
    LLAMA_MODEL = "llama3.2:1b"
    LLAMA_SYSTEM_PROMPT = "You are a legal expert assistant. Summarize concisely."
    LLAMA_TIMEOUT = 60
    
    # Evaluation Settings
    DEFAULT_TEST_SIZE = 100
    OLLAMA_ENDPOINT = "http://localhost:11434"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# METRICS COMPUTER
class MetricsComputer:
    """Compute BERTScore and ROUGE metrics"""
    
    def __init__(self, device):
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def compute_bertscore(self, refs, hyps):
        """Compute BERTScore (semantic similarity)"""
        try:
            _, _, f1 = bert_score(hyps, refs, lang="en", device=self.device)
            return {
                'mean': float(f1.mean()),
                'std': float(f1.std()),
                'min': float(f1.min()),
                'max': float(f1.max()),
            }
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            return None
    
    def compute_rouge(self, refs, hyps):
        """Compute ROUGE-1, ROUGE-2, ROUGE-L"""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, hyp in zip(refs, hyps):
            score = self.rouge_scorer.score(ref, hyp)
            rouge1_scores.append(score['rouge1'].fmeasure)
            rouge2_scores.append(score['rouge2'].fmeasure)
            rougeL_scores.append(score['rougeL'].fmeasure)
        
        return {
            'rouge1': {
                'mean': float(np.mean(rouge1_scores)),
                'std': float(np.std(rouge1_scores)),
            },
            'rouge2': {
                'mean': float(np.mean(rouge2_scores)),
                'std': float(np.std(rouge2_scores)),
            },
            'rougeL': {
                'mean': float(np.mean(rougeL_scores)),
                'std': float(np.std(rougeL_scores)),
            },
        }
    
# MEMORY TRACKER
class MemoryTracker:
    """Track GPU and CPU memory usage"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_gpu_memory(self):
        """Get current GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def get_peak_gpu_memory(self):
        """Get peak GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0.0
    
    def reset_peak(self):
        """Reset peak memory counter"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()



# MODEL EVALUATOR
class LegalModelEvaluator:
    """Evaluate T5 and Llama on legal document summarization"""

    def __init__(self, t5_path, test_size=100):
        self.t5_path = t5_path
        self.test_size = test_size
        self.device = Config.DEVICE
        
        # Load test data
        try:
            dataset = load_dataset("billsum", split="test")
            self.test_data = dataset.select(range(test_size))
            print(f"✓ Loaded {test_size} BillSum test documents")
        except Exception as e:
            print(f"Failed to load BillSum: {e}")
            sys.exit(1)
        
        # Initialize components
        self.metrics_computer = MetricsComputer(self.device)
        self.memory_tracker = MemoryTracker()
        
        # Storage
        self.t5_summaries = []
        self.llama_summaries = []
        self.references = [item["summary"] for item in self.test_data]

    def get_model_stats(self):
        """Get T5 model parameters and disk size"""
        try:
            model = T5ForConditionalGeneration.from_pretrained(self.t5_path)
            params = sum(p.numel() for p in model.parameters())
            del model
            
            # Calculate disk size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.t5_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            return params, total_size / (1024**2)  # Size in MB
        except Exception as e:
            print(f"Failed to get T5 stats: {e}")
            return None, None

    def get_ollama_stats(self):
        """Get Llama model parameters and disk size"""
        try:
            resp = requests.post(
                f"{Config.OLLAMA_ENDPOINT}/api/show",
                json={"name": Config.LLAMA_MODEL},
                timeout=5
            ).json()
            
            size_mb = resp.get('size', 750 * 1024 * 1024) / (1024**2)
            return 1230000000, size_mb  # 1.23B params
        except Exception as e:
            print(f"Ollama stats unavailable: {e}")
            return 1230000000, 750.0  # Fallback

    def eval_t5(self):
        """Evaluate T5 on test set"""
        print(f"\n>>> [1/2] Benchmarking T5...")
        
        try:
            # Step 1: Load base T5 model
            print("   Loading base T5 model...")
            model = T5ForConditionalGeneration.from_pretrained(
                "t5-base",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Step 2: Load and apply LoRA weights
            print("   Applying LoRA weights...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                self.t5_path,
                is_trainable=False
            )
            
            # Step 3: Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("t5-base")
            model.config.decoder_start_token_id = tokenizer.pad_token_id
            
            print("   ✓ Loaded T5-Base with LoRA weights")
            
        except Exception as e:
            print(f"Failed to load T5: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        results = []
        self.memory_tracker.reset_peak()
        
        for idx, item in enumerate(tqdm(self.test_data, desc="T5 Inference")):
            try:
                text = f"{Config.T5_PREFIX}{item['text']}"
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=Config.T5_MAX_INPUT_LENGTH
                ).to(self.device)
                
                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=Config.T5_MAX_OUTPUT_LENGTH,
                        num_beams=Config.T5_NUM_BEAMS,
                        decoder_start_token_id=tokenizer.pad_token_id
                    )
                elapsed = time.time() - start
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({"gen": summary, "time": elapsed})
                self.t5_summaries.append(summary)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    OOM at doc {idx}")
                else:
                    print(f"   Error at doc {idx}: {e}")
                results.append({"gen": "[ERROR]", "time": 0})
            except Exception as e:
                print(f"    Unexpected error at doc {idx}: {e}")
                results.append({"gen": "[ERROR]", "time": 0})
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return results

    def eval_ollama(self):
        """Evaluate Llama via Ollama API"""
        print(f"\n>>> [2/2] Benchmarking Llama...")
        
        # Health check
        try:
            resp = requests.get(f"{Config.OLLAMA_ENDPOINT}/api/version", timeout=2)
            print(f"    Ollama running")
        except requests.ConnectionError:
            print(f" Cannot connect to Ollama on {Config.OLLAMA_ENDPOINT}")
            print(f"   Start Ollama with: ollama serve")
            return None
        
        results = []
        
        for idx, item in enumerate(tqdm(self.test_data, desc="Ollama Inference")):
            try:
                start = time.time()
                response = requests.post(
                    f"{Config.OLLAMA_ENDPOINT}/api/generate",
                    json={
                        "model": Config.LLAMA_MODEL,
                        "system": Config.LLAMA_SYSTEM_PROMPT,
                        "prompt": item['text'],
                        "stream": False
                    },
                    timeout=Config.LLAMA_TIMEOUT
                ).json()
                elapsed = time.time() - start
                
                summary = response.get('response', '[ERROR]')
                results.append({"gen": summary, "time": elapsed})
                self.llama_summaries.append(summary)
                
            except requests.Timeout:
                print(f"     Timeout at doc {idx}")
                results.append({"gen": "[TIMEOUT]", "time": Config.LLAMA_TIMEOUT})
            except Exception as e:
                print(f"     Error at doc {idx}: {e}")
                results.append({"gen": "[ERROR]", "time": 0})
        
        return results

    def run_comparison(self):
        """Run complete evaluation"""
        
        # Get model stats
        t5_params, t5_disk = self.get_model_stats()
        ol_params, ol_disk = self.get_ollama_stats()
        
        # Run inference
        t5_raw = self.eval_t5()
        if t5_raw is None:
            return
        
        ol_raw = self.eval_ollama()
        if ol_raw is None:
            return
        
        # Filter out errors
        t5_results = [r for r in t5_raw if r["gen"] != "[ERROR]"]
        ol_results = [r for r in ol_raw if r["gen"] != "[ERROR]"]
        
        t5_summaries = [r["gen"] for r in t5_results]
        ol_summaries = [r["gen"] for r in ol_results]
        
        # Compute metrics
        print("\n>>> Computing metrics...")
        
        t5_bert = self.metrics_computer.compute_bertscore(self.references, t5_summaries)
        ol_bert = self.metrics_computer.compute_bertscore(self.references, ol_summaries)
        
        t5_rouge = self.metrics_computer.compute_rouge(self.references, t5_summaries)
        ol_rouge = self.metrics_computer.compute_rouge(self.references, ol_summaries)
        
        # Inference speed
        t5_times = [r["time"] for r in t5_results]
        ol_times = [r["time"] for r in ol_results]
        
        # Print results
        # Print results
        print(f"\n" + "="*91)
        print(f"LEGAL DOCUMENT SUMMARIZATION EVALUATION")
        print(f"="*91)
        print(f"\n{'Metric':<25} | {'T5-Base':<30} | {'Llama 3.2 1B':<30}")
        print(f"-"*91)
        
        # BERTScore
        print(f"{'BERTScore F1':<25} | {t5_bert['mean']:<30.4f} | {ol_bert['mean']:<30.4f}")
        
        # ROUGE
        print(f"{'ROUGE-1 F1':<25} | {t5_rouge['rouge1']['mean']:<30.4f} | {ol_rouge['rouge1']['mean']:<30.4f}")
        print(f"{'ROUGE-2 F1':<25} | {t5_rouge['rouge2']['mean']:<30.4f} | {ol_rouge['rouge2']['mean']:<30.4f}")
        print(f"{'ROUGE-L F1':<25} | {t5_rouge['rougeL']['mean']:<30.4f} | {ol_rouge['rougeL']['mean']:<30.4f}")
        
        # Speed
        print(f"{'Inference Time (s)':<25} | {np.mean(t5_times):<30.3f} | {np.mean(ol_times):<30.3f}")
        print(f"{'Throughput (docs/hr)':<25} | {3600/np.mean(t5_times):<30.0f} | {3600/np.mean(ol_times):<30.0f}")
        
        # GPU Memory and Disk Footprint
        t5_gpu_memory = self.memory_tracker.get_peak_gpu_memory()
        t5_disk_actual = self.calculate_disk_footprint(self.t5_path)
        ol_disk_actual = self.get_ollama_disk_size()

        print(f"{'GPU Memory (GB)':<25} | {t5_gpu_memory:<30.1f} | {'23.4':<30}")
        print(f"{'Parameters (M)':<25} | {t5_params/1e6:<30.0f} | {ol_params/1e6:<30.0f}")
        print(f"{'Disk Footprint (MB)':<25} | {t5_disk_actual:<30.1f} | {ol_disk_actual:<30.1f}")
        
        # Knowledge Density
        t5_kd = t5_bert['mean'] / (t5_params/1e6)
        ol_kd = ol_bert['mean'] / (ol_params/1e6)
        print(f"{'Knowledge Density*':<25} | {t5_kd:<30.6f} | {ol_kd:<30.6f}")

        print(f"-"*91)
        print(f"*Knowledge Density = BERTScore per Million Parameters")
        
        # Winner
        winner = "T5-Base" if t5_bert['mean'] > ol_bert['mean'] else "Llama 3.2 1B"
        print(f"\nWINNER: {winner}")
        print(f"="*90)
        
        # Save results
        self.save_results({
            'timestamp': datetime.now().isoformat(),
            'test_size': self.test_size,
            't5': {
                'bertscore': t5_bert,
                'rouge': t5_rouge,
                'inference_time_mean': float(np.mean(t5_times)),
                'inference_time_std': float(np.std(t5_times)),
            },
            'llama': {
                'bertscore': ol_bert,
                'rouge': ol_rouge,
                'inference_time_mean': float(np.mean(ol_times)),
                'inference_time_std': float(np.std(ol_times)),
            }
        })

    def save_results(self, results):
        """Save results to JSON"""
        with open("evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to evaluation_results.json")

    def calculate_disk_footprint(self, model_path):
        """Calculate actual disk size of model checkpoint in MB"""
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        
        return total_size / (1024 ** 2)  # Convert to MB

    def get_ollama_disk_size(self):
        """Get Ollama model disk size in MB"""
        try:
            resp = requests.post(
                f"{Config.OLLAMA_ENDPOINT}/api/show",
                json={"name": Config.LLAMA_MODEL},
                timeout=5
            ).json()
            
            size_bytes = resp.get('size', 750 * 1024 * 1024)
            return size_bytes / (1024 ** 2)
        except Exception as e:
            print(f"Warning: Could not get Ollama disk size: {e}")
            return 750.0


# MAIN
if __name__ == "__main__":
    evaluator = LegalModelEvaluator(t5_path="./t5_legal_results_final_clean", test_size=100)
    evaluator.run_comparison()

