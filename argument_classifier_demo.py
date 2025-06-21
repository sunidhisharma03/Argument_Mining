#!/usr/bin/env python3
"""
Argument Mining Classifier - Classroom Demo
============================================
This script demonstrates a trained BERT model that classifies legal text into:
- Non-Argument (0): Regular text that doesn't make an argument
- Premise (1): Supporting evidence or reasoning
- Conclusion (2): Final decision or judgment
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class ArgumentClassifier:
    def __init__(self, model_path="./legal-bert-argument-classifier"):
        """Initialize the argument classifier"""
        print("Loading trained model...")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("Please ensure the model files are in the correct directory.")
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            self.label_map = {0: "Non-Argument", 1: "Premise", 2: "Conclusion"}
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def predict(self, text):
        """Predict argument type for given text"""
        if not hasattr(self, 'model'):
            return None
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256
        )
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "text": text,
            "prediction": self.label_map[predicted_class],
            "confidence": confidence,
            "probabilities": {
                self.label_map[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def demo_examples(self):
        """Run demo with predefined examples"""
        examples = [
            "The Court finds that there has been a violation of Article 8 of the Convention.",
            "Therefore, the application must be dismissed as inadmissible.",
            "The applicant was detained without access to a lawyer for 48 hours.",
            "Article 6 of the Convention guarantees the right to a fair trial.",
            "The hearing was scheduled for Tuesday morning at 10 AM.",
            "It was raining heavily on the day of the incident."
        ]
        
        print("\n" + "="*60)
        print("🎯 ARGUMENT MINING CLASSIFIER DEMO")
        print("="*60)
        
        for i, example in enumerate(examples, 1):
            print(f"\n📝 Example {i}:")
            print(f"Text: \"{example}\"")
            
            result = self.predict(example)
            if result:
                colors = {"Non-Argument": "🔵", "Premise": "🟡", "Conclusion": "🟢"}
                print(f"Prediction: {colors[result['prediction']]} {result['prediction']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                print("Probabilities:")
                for label, prob in result['probabilities'].items():
                    bar_length = int(prob * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"  {label:12}: {bar} {prob:.3f}")
            
            print("-" * 50)

def main():
    print("🏛️ Legal Argument Mining with BERT")
    classifier = ArgumentClassifier()
    
    if hasattr(classifier, 'model'):
        classifier.demo_examples()
        
        # Interactive mode
        print("\n🔄 Interactive mode - Enter text to classify ('quit' to exit):")
        while True:
            try:
                user_input = input("\n📝 Enter text: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    result = classifier.predict(user_input)
                    if result:
                        print(f"🎯 Prediction: {result['prediction']}")
                        print(f"📊 Confidence: {result['confidence']:.3f}")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
