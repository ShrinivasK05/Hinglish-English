#scareBLEU code
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import MBartForConditionalGeneration, MBartTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import math
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import sacreBLEU for BLEU-4 evaluation
try:
    from sacrebleu import sentence_bleu, corpus_bleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("⚠️  Warning: sacreBLEU not installed. Install with: pip install sacrebleu")
    print("   Falling back to basic BLEU implementation")

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '.', text)
    return text.strip()

def clean_dataset(hinglish_texts, english_texts):
    """Filter and clean dataset"""
    cleaned_h, cleaned_e = [], []
    for h, e in zip(hinglish_texts, english_texts):
        h_clean = preprocess_text(h)
        e_clean = preprocess_text(e)
        if len(h_clean) > 2 and len(e_clean) > 2:
            if len(h_clean.split()) <= 50 and len(e_clean.split()) <= 50:
                cleaned_h.append(h_clean)
                cleaned_e.append(e_clean)
    print(f"Dataset cleaned: {len(hinglish_texts)} -> {len(cleaned_h)} samples")
    return cleaned_h, cleaned_e

def create_dataset_tensors(source_texts, target_texts, tokenizer, max_length=128):
    """Create dataset tensors without custom class"""
    print(f"   Tokenizing {len(source_texts)} samples...")

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for source, target in tqdm(zip(source_texts, target_texts), total=len(source_texts), desc="Tokenizing"):
        # Tokenize source
        source_encoded = tokenizer(
            str(source), max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        # Tokenize target
        target_encoded = tokenizer(
            str(target), max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        # Create labels (replace padding tokens with -100)
        labels = target_encoded['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # Store tensors
        all_input_ids.append(source_encoded['input_ids'].squeeze())
        all_attention_masks.append(source_encoded['attention_mask'].squeeze())
        all_labels.append(labels.squeeze())

    # Stack all tensors
    input_ids_tensor = torch.stack(all_input_ids)
    attention_mask_tensor = torch.stack(all_attention_masks)
    labels_tensor = torch.stack(all_labels)

    print(f"   Created tensors: {input_ids_tensor.shape}")

    # Create TensorDataset (built-in PyTorch)
    dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)
    return dataset

def initialize_model():
    """Initialize mBART model and tokenizer"""
    print("🚀 Initializing mBART model...")
    print("   This may take a few minutes (downloading ~2.3GB)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")

    try:
        print("📥 Loading tokenizer...")
        tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

        print("📥 Loading model...")
        model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')

        # Configure tokenizer
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "en_XX"

        # Move model to device
        print(f"📤 Moving model to {device}...")
        model.to(device)

        print("✅ Model initialized successfully!")
        return model, tokenizer, device

    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Try:")
        print("1. Check internet connection")
        print("2. pip install --upgrade transformers torch")
        print("3. Restart Python")
        raise

def load_dataset(file_path, max_samples=None):
    """Load dataset from CSV file"""
    print(f"📊 Loading dataset: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)
    print(f"   Found columns: {list(df.columns)}")

    # Handle column names for your dataset
    if 'Sentence' in df.columns and 'English_Translation' in df.columns:
        hinglish_col = 'Sentence'
        english_col = 'English_Translation'
    elif 'sentence' in df.columns and 'english_translation' in df.columns:
        hinglish_col = 'sentence'
        english_col = 'english_translation'
    else:
        raise ValueError(f"Expected 'Sentence'/'English_Translation' columns. Found: {list(df.columns)}")

    print(f"   Using columns: '{hinglish_col}' -> '{english_col}'")

    # Extract data
    hinglish_texts = df[hinglish_col].dropna().astype(str).tolist()
    english_texts = df[english_col].dropna().astype(str).tolist()

    print(f"   Raw samples: {len(hinglish_texts)}")

    # Limit samples if requested
    if max_samples and len(hinglish_texts) > max_samples:
        print(f"   Limiting to {max_samples} random samples")
        indices = np.random.choice(len(hinglish_texts), max_samples, replace=False)
        hinglish_texts = [hinglish_texts[i] for i in indices]
        english_texts = [english_texts[i] for i in indices]

    # Clean and return
    return clean_dataset(hinglish_texts, english_texts)

def train_one_stage(model, tokenizer, device, source_texts, target_texts,
                   stage_name, epochs=3, batch_size=4, lr=5e-5):
    """Train one stage of the model"""
    print(f"\n{'='*50}")
    print(f"🔄 {stage_name}")
    print(f"   Samples: {len(source_texts)}, Epochs: {epochs}, Batch: {batch_size}")
    print(f"{'='*50}")

    # Create dataset tensors
    dataset = create_dataset_tensors(source_texts, target_texts, tokenizer)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    model.train()
    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # Unpack batch (TensorDataset returns tuple)
            input_ids, attention_mask, labels = batch

            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Track loss
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Print epoch results
        avg_loss = epoch_loss / len(dataloader)
        print(f"   Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        total_loss += avg_loss

    final_avg_loss = total_loss / epochs
    print(f"✅ {stage_name} Complete! Final Average Loss: {final_avg_loss:.4f}")
    return final_avg_loss

def translate_text(model, tokenizer, device, hinglish_text, max_length=128):
    """Translate a single Hinglish text to English"""
    model.eval()
    text = preprocess_text(hinglish_text)

    if not text.strip():
        return ""

    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors='pt', max_length=max_length,
            padding=True, truncation=True
        ).to(device)

        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            max_length=max_length,
            min_length=3,
            num_beams=5,
            length_penalty=1.2,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3
        )

        translation = tokenizer.decode(generated[0], skip_special_tokens=True)
        return preprocess_text(translation)

def calculate_bleu_score(predictions, references):
    """Calculate BLEU-4 score using sacreBLEU"""
    if SACREBLEU_AVAILABLE:
        # Use sacreBLEU for BLEU-4 calculation
        print("📊 Calculating BLEU-4 score using sacreBLEU...")

        # Filter out empty predictions and references
        valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references)
                      if pred.strip() and ref.strip()]

        if not valid_pairs:
            return 0.0

        valid_predictions, valid_references = zip(*valid_pairs)

        # Calculate corpus BLEU-4 score
        bleu_score = corpus_bleu(valid_predictions, [valid_references])
        return bleu_score.score / 100.0  # Convert to 0-1 scale
    else:
        # Fallback to basic BLEU implementation
        print("📊 Calculating BLEU score using basic implementation...")

        def bleu_for_pair(reference, prediction):
            ref_words = reference.lower().split()
            pred_words = prediction.lower().split()

            if not pred_words:
                return 0.0

            # 1-gram precision
            ref_counter = Counter(ref_words)
            pred_counter = Counter(pred_words)
            matches = sum((ref_counter & pred_counter).values())
            precision = matches / len(pred_words)

            # Brevity penalty
            if len(pred_words) > len(ref_words):
                bp = 1.0
            else:
                bp = math.exp(1 - len(ref_words) / len(pred_words)) if pred_words else 0

            return bp * precision

        scores = []
        for pred, ref in zip(predictions, references):
            if pred.strip() and ref.strip():
                scores.append(bleu_for_pair(ref, pred))

        return sum(scores) / len(scores) if scores else 0.0

def evaluate_model(model, tokenizer, device, hinglish_texts, english_texts):
    """Evaluate model and calculate BLEU-4 score"""
    print(f"\n📊 Evaluating model on {len(hinglish_texts)} samples...")

    # Generate predictions
    predictions = []
    for text in tqdm(hinglish_texts, desc="Translating"):
        prediction = translate_text(model, tokenizer, device, text)
        predictions.append(prediction)

    # Calculate BLEU-4 score
    bleu_score = calculate_bleu_score(predictions, english_texts)

    if SACREBLEU_AVAILABLE:
        print(f"🎯 BLEU-4 Score (sacreBLEU): {bleu_score:.4f}")
    else:
        print(f"🎯 BLEU Score (basic): {bleu_score:.4f}")

    # Show sample translations
    print(f"\n📝 Sample Translations:")
    print("-" * 60)
    for i in range(min(5, len(hinglish_texts))):
        print(f"Input:     {hinglish_texts[i]}")
        print(f"Reference: {english_texts[i]}")
        print(f"Generated: {predictions[i]}")
        print("-" * 60)

    return bleu_score

def interactive_translation(model, tokenizer, device):
    """Interactive translation mode"""
    print(f"\n🗣  Interactive Translation Mode")
    print("Enter Hinglish text to translate (type 'quit' to exit)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n💬 Hinglish: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif not user_input:
                continue

            translation = translate_text(model, tokenizer, device, user_input)
            print(f"🎯 English: {translation}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main function"""
    print("🌟 Enhanced Hinglish to English Translator")
    print("   2-Stage Training with mBART")
    print("=" * 55)

    # Check sacreBLEU availability
    if not SACREBLEU_AVAILABLE:
        print("🔧 For accurate BLEU-4 scores, install sacreBLEU:")
        print("   pip install sacrebleu")
        print("-" * 55)

    # Step 1: Initialize model
    try:
        model, tokenizer, device = initialize_model()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Step 2: Load dataset
    print(f"\n📂 Dataset Setup")

    # Check for your dataset
    default_path = "hinglish_phinc.csv"
    if os.path.exists(default_path):
        use_default = input(f"Found '{default_path}'. Use this dataset? (y/n): ").strip().lower()
        if use_default == 'y':
            dataset_path = default_path
        else:
            dataset_path = input("Enter CSV file path (or 'demo'): ").strip()
    else:
        dataset_path = input("Enter CSV file path (or 'demo'): ").strip()

    # Load data
    if dataset_path.lower() == 'demo':
        # Demo data
        hinglish_texts = [
            "Main apne ghar ja raha hun", "Yeh movie bahut acchi hai",
            "Kya aap hindi samajh sakte hain", "Mujhe pizza khana pasand hai",
            "Kal main office nahi jaunga", "Aaj weather kaisa hai",
            "Main shopping kar raha hun", "Yeh book interesting hai",
            "Tumhara naam kya hai", "Main coffee peena chahta hun",
            "Aaj main school jaunga", "Yeh gaana bahut sundar hai",
            "Main apne friends ke saath party kar raha hun", "Cricket match kaun jeeta",
            "Mujhe ice cream khana hai", "Tum kya kar rahe ho"
        ]
        english_texts = [
            "I am going to my home", "This movie is very good",
            "Can you understand Hindi", "I like to eat pizza",
            "Tomorrow I will not go to office", "How is the weather today",
            "I am shopping", "This book is interesting",
            "What is your name", "I want to drink coffee",
            "Today I will go to school", "This song is very beautiful",
            "I am partying with my friends", "Who won the cricket match",
            "I want to eat ice cream", "What are you doing"
        ]
        print(f"✅ Demo dataset loaded: {len(hinglish_texts)} samples")
    else:
        try:
            # Handle large dataset
            if 'phinc' in dataset_path.lower():
                print("📊 Large PHINC dataset detected!")
                print("Memory management options:")
                print("1. Full dataset (requires 8GB+ RAM, best results)")
                print("2. 10000 samples (recommended for 4-8GB RAM)")
                print("3. 2000 samples (for systems with <4GB RAM)")

                choice = input("Choose option (1/2/3): ").strip()
                max_samples = None
                if choice == '2':
                    max_samples = 10000
                elif choice == '3':
                    max_samples = 2000

                if max_samples:
                    print(f"Will use {max_samples} randomly selected samples")
            else:
                max_samples = None

            hinglish_texts, english_texts = load_dataset(dataset_path, max_samples)

        except Exception as e:
            print(f"❌ Dataset loading error: {e}")
            return

    print(f"✅ Final dataset size: {len(hinglish_texts)} samples")

    # Step 3: Training configuration
    print(f"\n⚙  Training Configuration")
    dataset_size = len(hinglish_texts)

    if dataset_size > 5000:
        print("🎯 Large dataset - recommended settings:")
        print("1. Full Training - Best BLEU scores (~45-90 min)")
        print("2. Quick Training - Faster training (~15-30 min)")
    else:
        print("1. Full Training - Better results (~20-45 min)")
        print("2. Quick Training - Faster training (~5-15 min)")

    choice = input("Choose training mode (1/2): ").strip()
    is_full_training = (choice == '1')

    # Determine training parameters
    if torch.cuda.is_available():
        batch_size = 6 if dataset_size > 1000 else 4
    else:
        batch_size = 4 if dataset_size > 1000 else 2

    if is_full_training:
        stage1_epochs = 4 if dataset_size > 1000 else 3
        stage2_epochs = 6 if dataset_size > 1000 else 5
    else:
        stage1_epochs = 2
        stage2_epochs = 3

    print(f"📊 Training parameters:")
    print(f"   Batch size: {batch_size}")
    print(f"   Stage 1 epochs: {stage1_epochs}")
    print(f"   Stage 2 epochs: {stage2_epochs}")

    # Step 4: Split dataset
    split_ratio = 0.8
    split_idx = int(split_ratio * len(hinglish_texts))

    train_hinglish = hinglish_texts[:split_idx]
    train_english = english_texts[:split_idx]
    test_hinglish = hinglish_texts[split_idx:]
    test_english = english_texts[split_idx:]

    # Ensure we have test data
    if len(test_hinglish) == 0:
        test_hinglish = train_hinglish[:min(5, len(train_hinglish))]
        test_english = train_english[:min(5, len(train_english))]

    print(f"📊 Data split: {len(train_hinglish)} training, {len(test_hinglish)} testing")

    # Step 5: 2-Stage Training
    print(f"\n🎓 Starting 2-Stage Training")

    # Stage 1: English to Hinglish (Reverse Translation)
    stage1_loss = train_one_stage(
        model, tokenizer, device,
        train_english, train_hinglish,
        "STAGE 1: English → Hinglish",
        epochs=stage1_epochs,
        batch_size=batch_size,
        lr=5e-5
    )

    # Stage 2: Hinglish to English (Main Task)
    stage2_loss = train_one_stage(
        model, tokenizer, device,
        train_hinglish, train_english,
        "STAGE 2: Hinglish → English",
        epochs=stage2_epochs,
        batch_size=batch_size,
        lr=3e-5
    )

    print(f"\n🎉 2-Stage Training Complete!")
    print(f"📈 Stage 1 Loss: {stage1_loss:.4f}")
    print(f"📈 Stage 2 Loss: {stage2_loss:.4f}")

    # Step 6: Evaluation
    bleu_score = evaluate_model(model, tokenizer, device, test_hinglish, test_english)

    # Step 7: Interactive mode
    if SACREBLEU_AVAILABLE:
        print(f"\nFinal BLEU-4 Score (sacreBLEU): {bleu_score:.4f}")
    else:
        print(f"\nFinal BLEU Score: {bleu_score:.4f}")

    if input("\nTry interactive translation? (y/n): ").strip().lower() == 'y':
        interactive_translation(model, tokenizer, device)

    # Step 8: Save model (optional)
    if input("\nSave trained model? (y/n): ").strip().lower() == 'y':
        save_path = input("Save path (default: ./hinglish_model): ").strip() or "./hinglish_model"
        try:
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✅ Model saved to {save_path}")
        except Exception as e:
            print(f"❌ Save failed: {e}")

    if SACREBLEU_AVAILABLE:
        print(f"\n🎉 Training Complete! Final BLEU-4 Score (sacreBLEU): {bleu_score:.4f}")
    else:
        print(f"\n🎉 Training Complete! Final BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    main()