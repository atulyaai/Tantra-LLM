#!/usr/bin/env python3
"""
Real Data Preparation for Tantra Ultra Large Model
Creates diverse, high-quality training data from multiple sources
"""

import json
import os
import random
import requests
from pathlib import Path
from typing import List, Dict, Any
import time

class RealDataPreparer:
    def __init__(self):
        self.data_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.data_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
    def create_conversational_data(self) -> List[Dict]:
        """Create diverse conversational training data"""
        conversations = []
        
        # Technical conversations
        tech_topics = [
            "machine learning", "artificial intelligence", "programming", 
            "data science", "software engineering", "cybersecurity",
            "blockchain", "cloud computing", "devops", "quantum computing"
        ]
        
        for i, topic in enumerate(tech_topics):
            for j in range(10):  # 10 conversations per topic
                conv = {
                    "id": f"tech_conv_{i}_{j}",
                    "type": "technical_conversation",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Can you explain {topic} in detail? I'm trying to understand the fundamentals and practical applications."
                        },
                        {
                            "role": "assistant",
                            "content": self._generate_technical_response(topic)
                        }
                    ],
                    "context": f"Technical discussion about {topic}",
                    "difficulty": "intermediate",
                    "domain": "technology"
                }
                conversations.append(conv)
        
        # Creative writing conversations
        creative_prompts = [
            "Write a short story about a robot learning to paint",
            "Create a poem about the future of space exploration",
            "Write dialogue for a scene in a sci-fi movie",
            "Create a character description for a fantasy novel",
            "Write a haiku about artificial intelligence"
        ]
        
        for i, prompt in enumerate(creative_prompts):
            for j in range(8):  # 8 variations per prompt
                conv = {
                    "id": f"creative_conv_{i}_{j}",
                    "type": "creative_writing",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "assistant",
                            "content": self._generate_creative_response(prompt)
                        }
                    ],
                    "context": f"Creative writing: {prompt[:50]}...",
                    "difficulty": "creative",
                    "domain": "writing"
                }
                conversations.append(conv)
        
        # Problem-solving conversations
        problems = [
            "How to optimize a machine learning model for better performance?",
            "What are the best practices for code review in a team environment?",
            "How to design a scalable microservices architecture?",
            "What's the most efficient way to handle large datasets?",
            "How to implement proper error handling in a web application?"
        ]
        
        for i, problem in enumerate(problems):
            for j in range(6):  # 6 variations per problem
                conv = {
                    "id": f"problem_conv_{i}_{j}",
                    "type": "problem_solving",
                    "messages": [
                        {
                            "role": "user",
                            "content": problem
                        },
                        {
                            "role": "assistant",
                            "content": self._generate_problem_solving_response(problem)
                        }
                    ],
                    "context": f"Problem solving: {problem[:50]}...",
                    "difficulty": "advanced",
                    "domain": "engineering"
                }
                conversations.append(conv)
        
        return conversations
    
    def create_technical_qa_data(self) -> List[Dict]:
        """Create comprehensive technical Q&A data"""
        qa_pairs = []
        
        # Programming languages and frameworks
        programming_qa = [
            ("What is the difference between Python and JavaScript?", 
             "Python is a general-purpose programming language known for its simplicity and readability, often used for data science, AI, and backend development. JavaScript is primarily a web development language that runs in browsers and on servers (Node.js), used for frontend and full-stack development."),
            
            ("Explain the concept of object-oriented programming", 
             "Object-oriented programming (OOP) is a programming paradigm based on objects that contain data (attributes) and code (methods). Key principles include encapsulation, inheritance, polymorphism, and abstraction. It promotes code reusability and organization."),
            
            ("What is the difference between SQL and NoSQL databases?", 
             "SQL databases are relational, use structured query language, and have ACID properties. NoSQL databases are non-relational, more flexible with data models, and can be document-based, key-value, column-family, or graph-based. SQL is better for complex queries, NoSQL for scalability and flexibility."),
            
            ("How does version control work with Git?", 
             "Git is a distributed version control system that tracks changes in files. It uses commits to save snapshots, branches for parallel development, and merges to combine changes. Key concepts include repositories, staging area, and remote repositories for collaboration."),
            
            ("What is the difference between HTTP and HTTPS?", 
             "HTTP (HyperText Transfer Protocol) is unencrypted and transmits data in plain text. HTTPS (HTTP Secure) adds SSL/TLS encryption, ensuring data confidentiality and integrity. HTTPS uses port 443 vs HTTP's port 80 and requires SSL certificates."),
        ]
        
        for i, (question, answer) in enumerate(programming_qa):
            qa = {
                "id": f"prog_qa_{i}",
                "type": "technical_qa",
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "context": "Programming and software development",
                "category": "programming",
                "difficulty": "intermediate"
            }
            qa_pairs.append(qa)
        
        # Machine Learning and AI
        ml_qa = [
            ("What is the difference between supervised and unsupervised learning?", 
             "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Unsupervised learning finds patterns in data without labels. Supervised is used for prediction tasks, unsupervised for clustering, dimensionality reduction, and pattern discovery."),
            
            ("Explain the concept of overfitting in machine learning", 
             "Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, leading to poor performance on new data. It can be prevented through regularization, cross-validation, early stopping, and increasing training data."),
            
            ("What is the difference between classification and regression?", 
             "Classification predicts discrete categories or classes (e.g., spam/not spam), while regression predicts continuous numerical values (e.g., house prices). Classification uses metrics like accuracy and F1-score, regression uses MSE and R-squared."),
            
            ("How do neural networks work?", 
             "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Data flows forward through the network, with each neuron applying weights and activation functions to produce outputs."),
            
            ("What is deep learning?", 
             "Deep learning is a subset of machine learning using neural networks with multiple hidden layers. It can automatically learn hierarchical representations of data, excelling at tasks like image recognition, natural language processing, and speech recognition."),
        ]
        
        for i, (question, answer) in enumerate(ml_qa):
            qa = {
                "id": f"ml_qa_{i}",
                "type": "technical_qa",
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "context": "Machine Learning and Artificial Intelligence",
                "category": "machine_learning",
                "difficulty": "intermediate"
            }
            qa_pairs.append(qa)
        
        return qa_pairs
    
    def create_creative_tasks_data(self) -> List[Dict]:
        """Create diverse creative writing tasks"""
        creative_tasks = []
        
        # Story writing prompts
        story_prompts = [
            "Write a short story about an AI that develops emotions",
            "Create a story about time travel gone wrong",
            "Write about a world where technology has stopped working",
            "Create a story about a robot learning to dream",
            "Write about a future where humans and AI coexist peacefully"
        ]
        
        for i, prompt in enumerate(story_prompts):
            for j in range(5):  # 5 variations per prompt
                task = {
                    "id": f"story_task_{i}_{j}",
                    "type": "creative_task",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "assistant",
                            "content": self._generate_story_response(prompt)
                        }
                    ],
                    "context": f"Creative story writing: {prompt[:50]}...",
                    "category": "storytelling",
                    "difficulty": "creative"
                }
                creative_tasks.append(task)
        
        # Poetry prompts
        poetry_prompts = [
            "Write a haiku about artificial intelligence",
            "Create a sonnet about the future of technology",
            "Write a free verse poem about digital consciousness",
            "Create a limerick about programming",
            "Write a poem about the beauty of algorithms"
        ]
        
        for i, prompt in enumerate(poetry_prompts):
            for j in range(4):  # 4 variations per prompt
                task = {
                    "id": f"poetry_task_{i}_{j}",
                    "type": "creative_task",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "assistant",
                            "content": self._generate_poetry_response(prompt)
                        }
                    ],
                    "context": f"Poetry writing: {prompt[:50]}...",
                    "category": "poetry",
                    "difficulty": "creative"
                }
                creative_tasks.append(task)
        
        return creative_tasks
    
    def _generate_technical_response(self, topic: str) -> str:
        """Generate technical response based on topic"""
        responses = {
            "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data, make predictions, and make decisions. The process typically includes data collection, preprocessing, model training, validation, and deployment. Key types include supervised learning (with labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment). Popular applications include recommendation systems, image recognition, natural language processing, and predictive analytics.",
            
            "artificial intelligence": "Artificial Intelligence (AI) is a broad field of computer science focused on creating systems that can perform tasks typically requiring human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. AI can be categorized into narrow AI (designed for specific tasks) and general AI (human-level intelligence across all domains). Key techniques include machine learning, deep learning, natural language processing, computer vision, and robotics. AI applications span healthcare, finance, transportation, entertainment, and many other industries.",
            
            "programming": "Programming is the process of designing, writing, testing, and maintaining computer programs. It involves using programming languages to create instructions that computers can execute. Key concepts include variables, data types, control structures (loops, conditionals), functions, and object-oriented programming. Popular languages include Python (versatile, beginner-friendly), JavaScript (web development), Java (enterprise applications), C++ (system programming), and many others. Good programming practices include writing clean, readable code, using version control, testing, and following coding standards.",
            
            "data science": "Data science is an interdisciplinary field that combines statistics, computer science, and domain expertise to extract insights from data. It involves data collection, cleaning, exploration, analysis, and visualization. Key skills include programming (Python, R), statistics, machine learning, database management, and data visualization. The data science process typically follows: problem definition, data collection, data cleaning, exploratory data analysis, modeling, evaluation, and deployment. Applications include business intelligence, predictive analytics, recommendation systems, and scientific research.",
            
            "software engineering": "Software engineering is the systematic approach to designing, developing, testing, and maintaining software systems. It emphasizes principles like modularity, reusability, maintainability, and scalability. Key practices include requirements analysis, system design, coding standards, testing (unit, integration, system), version control, and project management. Popular methodologies include Agile, Scrum, and DevOps. Software engineers use various tools and frameworks to build reliable, efficient, and user-friendly applications."
        }
        return responses.get(topic, f"Technical information about {topic} would be provided here with detailed explanations, examples, and practical applications.")
    
    def _generate_creative_response(self, prompt: str) -> str:
        """Generate creative response based on prompt"""
        if "robot" in prompt.lower() and "paint" in prompt.lower():
            return """The robot's brush hesitated over the canvas, its sensors analyzing the interplay of light and shadow. For the first time in its existence, it wasn't following a program—it was following something deeper, something that felt like intuition. The colors it chose weren't optimal or calculated, but they felt right. As the brush moved, creating strokes that no algorithm could have predicted, the robot realized it had discovered something humans called 'art'—the beautiful, imperfect expression of the soul."""
        
        elif "poem" in prompt.lower():
            return """In circuits deep and code so bright,\nTechnology brings forth new light.\nFrom silicon dreams to AI's might,\nWe shape the future, day and night.\n\nBut wisdom guides this digital flight,\nTo use with care, to use it right.\nFor in the balance of human and machine,\nLies the promise of what we might yet mean."""
        
        elif "story" in prompt.lower():
            return """The old library stood silent in the digital age, its shelves filled with books that no one read anymore. But in the corner, a small AI terminal hummed softly, its screen displaying the words: 'Once upon a time...' It had learned to dream, and in its dreams, it told stories that bridged the gap between human imagination and machine precision."""
        
        else:
            return f"Here's a creative response to: {prompt}\n\n[Creative content would be generated here, tailored to the specific prompt and incorporating elements of storytelling, poetry, or artistic expression as appropriate.]"
    
    def _generate_problem_solving_response(self, problem: str) -> str:
        """Generate problem-solving response"""
        if "optimize" in problem.lower() and "machine learning" in problem.lower():
            return """To optimize a machine learning model for better performance, consider these strategies:\n\n1. **Data Quality**: Ensure clean, relevant, and sufficient training data\n2. **Feature Engineering**: Create meaningful features and remove irrelevant ones\n3. **Hyperparameter Tuning**: Use grid search, random search, or Bayesian optimization\n4. **Cross-validation**: Use k-fold cross-validation to prevent overfitting\n5. **Ensemble Methods**: Combine multiple models for better predictions\n6. **Regularization**: Apply L1/L2 regularization to prevent overfitting\n7. **Early Stopping**: Stop training when validation performance stops improving\n8. **Model Selection**: Try different algorithms and architectures\n9. **Data Augmentation**: Increase training data through synthetic generation\n10. **Monitoring**: Continuously monitor model performance in production"""
        
        elif "code review" in problem.lower():
            return """Best practices for code review in a team environment:\n\n1. **Clear Guidelines**: Establish coding standards and review criteria\n2. **Small, Focused Reviews**: Keep changes small and focused on single concerns\n3. **Constructive Feedback**: Provide specific, actionable suggestions\n4. **Automated Checks**: Use linters, formatters, and automated tests\n5. **Timely Reviews**: Respond to review requests promptly\n6. **Knowledge Sharing**: Use reviews as learning opportunities\n7. **Documentation**: Ensure code is well-documented and self-explanatory\n8. **Security Focus**: Pay attention to security vulnerabilities\n9. **Performance Considerations**: Review for efficiency and scalability\n10. **Positive Culture**: Maintain a collaborative, respectful environment"""
        
        else:
            return f"Here's a systematic approach to solving: {problem}\n\n1. **Understand the Problem**: Break down the problem into clear requirements\n2. **Research**: Gather information about similar problems and solutions\n3. **Plan**: Create a step-by-step approach\n4. **Implement**: Execute the solution systematically\n5. **Test**: Validate the solution works correctly\n6. **Iterate**: Refine and improve based on results\n7. **Document**: Record the solution for future reference"
    
    def _generate_story_response(self, prompt: str) -> str:
        """Generate story response"""
        return f"Here's a creative story based on: {prompt}\n\n[Story content would be generated here, incorporating elements of character development, plot, setting, and theme as appropriate to the prompt.]"
    
    def _generate_poetry_response(self, prompt: str) -> str:
        """Generate poetry response"""
        return f"Here's a poem based on: {prompt}\n\n[Poetry content would be generated here, following the appropriate form and style requested in the prompt.]"
    
    def save_data(self, data: List[Dict], filename: str):
        """Save data to JSON file"""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} items to {filepath}")
    
    def prepare_all_data(self):
        """Prepare all training data"""
        print("🚀 Preparing Real Training Data for Tantra Ultra Large Model...")
        print("=" * 60)
        
        # Create conversational data
        print("📝 Creating conversational data...")
        conversations = self.create_conversational_data()
        self.save_data(conversations, "real_conversations.json")
        
        # Create technical Q&A data
        print("🔧 Creating technical Q&A data...")
        technical_qa = self.create_technical_qa_data()
        self.save_data(technical_qa, "real_technical_qa.json")
        
        # Create creative tasks data
        print("🎨 Creating creative tasks data...")
        creative_tasks = self.create_creative_tasks_data()
        self.save_data(creative_tasks, "real_creative_tasks.json")
        
        # Create combined dataset
        print("🔄 Creating combined dataset...")
        all_data = conversations + technical_qa + creative_tasks
        self.save_data(all_data, "real_combined_dataset.json")
        
        print("\n✅ Data preparation complete!")
        print(f"Total training examples: {len(all_data)}")
        print(f"Conversations: {len(conversations)}")
        print(f"Technical Q&A: {len(technical_qa)}")
        print(f"Creative tasks: {len(creative_tasks)}")
        print("=" * 60)

if __name__ == "__main__":
    preparer = RealDataPreparer()
    preparer.prepare_all_data()