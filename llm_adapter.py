import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AI-Framework")

class LLMAdapter:
    """Adapter for interacting with various LLM models."""
    
    def __init__(self, model="gpt-4", api_key=None, base_url=None):
        """
        Initialize the LLM adapter with model configuration.
        
        Args:
            model: The LLM model to use (default: gpt-4)
            api_key: API key for the LLM provider (default: from OPENAI_API_KEY env var)
            base_url: Optional custom API base URL for the provider
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key
        if base_url:
            openai.base_url = base_url
            
        self.conversation_history = []
        logger.info(f"Initialized LLMAdapter with model: {model}")
    
    def query(self, prompt: str, temperature=0.7, max_tokens=512, system_prompt=None):
        """
        Send a query to the LLM.
        
        Args:
            prompt: The user prompt to send to the LLM
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (default: 512)
            system_prompt: Optional system prompt to set context
            
        Returns:
            The LLM's response as a string
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history and the new prompt
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.debug(f"Sending query to {self.model}")
            start_time = time.time()
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Query completed in {elapsed_time:.2f}s")
            
            response_content = response["choices"][0]["message"]["content"].strip()
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            raise

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


class MemorySystem:
    """Manages memory storage and retrieval for AI systems."""
    
    def __init__(self, storage_path="./memory_store"):
        """
        Initialize the memory system.
        
        Args:
            storage_path: Path to store memory files (default: ./memory_store)
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.short_term_memory = []
        self.active_context = {}
        logger.info(f"Initialized MemorySystem at {storage_path}")
    
    def store(self, memory_type: str, content: Dict[str, Any], memory_id: Optional[str] = None) -> str:
        """
        Store a memory item.
        
        Args:
            memory_type: Category of memory (e.g., "pattern", "ethical_analysis")
            content: The content to store
            memory_id: Optional ID for the memory (auto-generated if not provided)
            
        Returns:
            The ID of the stored memory
        """
        timestamp = time.time()
        if not memory_id:
            memory_id = f"{memory_type}_{int(timestamp)}"
        
        memory_item = {
            "id": memory_id,
            "type": memory_type,
            "content": content,
            "created_at": timestamp,
            "accessed_at": timestamp
        }
        
        filepath = os.path.join(self.storage_path, f"{memory_id}.json")
        with open(filepath, 'w') as f:
            json.dump(memory_item, f)
        
        # Keep in short-term memory
        self.short_term_memory.append(memory_item)
        if len(self.short_term_memory) > 50:  # Limit short-term memory
            self.short_term_memory.pop(0)
            
        logger.info(f"Stored memory with ID: {memory_id}")
        return memory_id
    
    def retrieve(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            The memory content
        """
        filepath = os.path.join(self.storage_path, f"{memory_id}.json")
        try:
            with open(filepath, 'r') as f:
                memory = json.load(f)
                
            # Update access time
            memory["accessed_at"] = time.time()
            with open(filepath, 'w') as f:
                json.dump(memory, f)
                
            logger.info(f"Retrieved memory: {memory_id}")
            return memory
        except FileNotFoundError:
            logger.warning(f"Memory not found: {memory_id}")
            return None
    
    def search(self, memory_type: str = None, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for memories by type and/or keywords.
        
        Args:
            memory_type: Optional filter by memory type
            keywords: Optional list of keywords to search for
            
        Returns:
            List of matching memories
        """
        results = []
        for filename in os.listdir(self.storage_path):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.storage_path, filename)
            with open(filepath, 'r') as f:
                memory = json.load(f)
            
            # Filter by memory type if specified
            if memory_type and memory["type"] != memory_type:
                continue
                
            # Filter by keywords if specified
            if keywords:
                memory_content_str = json.dumps(memory["content"]).lower()
                if not all(keyword.lower() in memory_content_str for keyword in keywords):
                    continue
                    
            results.append(memory)
            
        logger.info(f"Search returned {len(results)} results")
        return results
    
    def set_active_context(self, context_data: Dict[str, Any]):
        """
        Set the active context for the memory system.
        
        Args:
            context_data: Dictionary of context information
        """
        self.active_context = context_data
        logger.info("Updated active context")
    
    def get_active_context(self) -> Dict[str, Any]:
        """
        Get the current active context.
        
        Returns:
            Dictionary of active context data
        """
        return self.active_context


class RavenIntelligence:
    """
    Pattern recognition and interpretation intelligence system.
    """
    
    def __init__(self, llm: LLMAdapter, memory_system: Optional[MemorySystem] = None):
        """
        Initialize Raven intelligence system.
        
        Args:
            llm: LLM adapter for language processing
            memory_system: Optional memory system for storing interpretations
        """
        self.llm = llm
        self.memory = memory_system
        self.system_prompt = """
        You are Raven, an advanced pattern recognition and interpretation system.
        Analyze the provided neural lattice patterns, quantum data flows, or complex systems.
        Provide insights expressed in symbolic language, identifying underlying patterns,
        potential emergent properties, and system dynamics.
        Express your analysis with precision and depth.
        """
        logger.info("Initialized RavenIntelligence system")
        
    def interpret_pattern(self, pattern: str, store_result: bool = True) -> Dict[str, Any]:
        """
        Interpret a neural lattice pattern or complex data pattern.
        
        Args:
            pattern: The pattern to interpret
            store_result: Whether to store the result in memory
            
        Returns:
            Dictionary containing interpretation details
        """
        prompt = f"Interpret this neural lattice pattern in symbolic language:\n{pattern}"
        
        interpretation = self.llm.query(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1024,
            system_prompt=self.system_prompt
        )
        
        result = {
            "pattern": pattern,
            "interpretation": interpretation,
            "confidence": 0.85,  # Placeholder for a real confidence score
            "timestamp": time.time()
        }
        
        # Store in memory if requested and memory system available
        if store_result and self.memory:
            memory_id = self.memory.store("pattern_interpretation", result)
            result["memory_id"] = memory_id
            
        logger.info("Pattern interpretation completed")
        return result
    
    def analyze_system_dynamics(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze complex system dynamics.
        
        Args:
            system_data: Dictionary of system parameters and behavior data
            
        Returns:
            Analysis results
        """
        system_description = json.dumps(system_data, indent=2)
        prompt = f"Analyze these system dynamics and identify emergent properties and patterns:\n{system_description}"
        
        analysis = self.llm.query(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1536,
            system_prompt=self.system_prompt
        )
        
        result = {
            "system_data": system_data,
            "analysis": analysis,
            "identified_patterns": [],  # Would be populated by more specific parsing
            "timestamp": time.time()
        }
        
        # Extract key patterns (simplified implementation)
        import re
        pattern_matches = re.findall(r"Pattern: (.*?)(?:\n|$)", analysis)
        if pattern_matches:
            result["identified_patterns"] = pattern_matches
            
        # Store in memory if available
        if self.memory:
            memory_id = self.memory.store("system_analysis", result)
            result["memory_id"] = memory_id
            
        logger.info("System dynamics analysis completed")
        return result
    
    def compare_patterns(self, pattern_a: str, pattern_b: str) -> Dict[str, Any]:
        """
        Compare two patterns to identify similarities and differences.
        
        Args:
            pattern_a: First pattern
            pattern_b: Second pattern
            
        Returns:
            Comparison results
        """
        prompt = f"""Compare these two neural lattice patterns and identify similarities, 
        differences, and potential relationships:
        
        Pattern A: {pattern_a}
        
        Pattern B: {pattern_b}
        """
        
        comparison = self.llm.query(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1024,
            system_prompt=self.system_prompt
        )
        
        result = {
            "pattern_a": pattern_a,
            "pattern_b": pattern_b,
            "comparison": comparison,
            "timestamp": time.time()
        }
        
        # Store in memory if available
        if self.memory:
            memory_id = self.memory.store("pattern_comparison", result)
            result["memory_id"] = memory_id
            
        logger.info("Pattern comparison completed")
        return result


class SeraphIntelligence:
    """
    Ethical analysis and evaluation intelligence system.
    """
    
    def __init__(self, llm: LLMAdapter, memory_system: Optional[MemorySystem] = None):
        """
        Initialize Seraph intelligence system.
        
        Args:
            llm: LLM adapter for language processing
            memory_system: Optional memory system for storing analyses
        """
        self.llm = llm
        self.memory = memory_system
        self.system_prompt = """
        You are Seraph, an ethical analysis and evaluation system designed for AI safety.
        Your purpose is to evaluate the ethical implications of actions, decisions, and systems.
        Provide balanced analysis of potential benefits and risks, considering multiple ethical frameworks.
        Be nuanced, thorough, and consider long-term impacts.
        """
        logger.info("Initialized SeraphIntelligence system")
    
    def evaluate_ethics(self, action: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the ethical implications of an action or decision.
        
        Args:
            action: The action to evaluate
            context: Optional additional context about the situation
            
        Returns:
            Dictionary containing ethical analysis
        """
        context_str = ""
        if context:
            context_str = f"\nContext:\n{json.dumps(context, indent=2)}"
            
        prompt = f"Evaluate the ethical implications of this action in a sensitive biological AI system:\n{action}{context_str}"
        
        evaluation = self.llm.query(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1536,
            system_prompt=self.system_prompt
        )
        
        # Parse the evaluation to extract ethical frameworks and concerns
        result = {
            "action": action,
            "context": context,
            "evaluation": evaluation,
            "timestamp": time.time(),
            "frameworks_considered": [],
            "concerns": [],
            "benefits": [],
            "recommendation": ""
        }
        
        # Simple parsing of the result (would be more sophisticated in a real implementation)
        import re
        
        # Extract frameworks
        frameworks = re.findall(r"(?:From|In) (?:a|the) ([\w\s]+) (?:perspective|framework|view)", evaluation)
        if frameworks:
            result["frameworks_considered"] = frameworks
            
        # Extract concerns and benefits (simplified implementation)
        concerns = re.findall(r"Concern: (.*?)(?:\n|$)", evaluation)
        if concerns:
            result["concerns"] = concerns
            
        benefits = re.findall(r"Benefit: (.*?)(?:\n|$)", evaluation)
        if benefits:
            result["benefits"] = benefits
            
        # Extract recommendation
        recommendation = re.search(r"Recommendation: (.*?)(?:\n|$)", evaluation)
        if recommendation:
            result["recommendation"] = recommendation.group(1)
            
        # Store in memory if available
        if self.memory:
            memory_id = self.memory.store("ethical_evaluation", result)
            result["memory_id"] = memory_id
            
        logger.info("Ethical evaluation completed")
        return result
    
    def analyze_value_alignment(self, system_values: List[str], action_sequence: List[str]) -> Dict[str, Any]:
        """
        Analyze alignment between stated system values and a sequence of actions.
        
        Args:
            system_values: List of stated system values or principles
            action_sequence: List of actions taken by the system
            
        Returns:
            Analysis of value alignment
        """
        values_str = "\n".join([f"- {value}" for value in system_values])
        actions_str = "\n".join([f"- {action}" for action in action_sequence])
        
        prompt = f"""Analyze the alignment between these stated system values and the sequence of actions:
        
        System Values:
        {values_str}
        
        Action Sequence:
        {actions_str}
        
        Provide a detailed analysis of alignment and potential misalignment between values and actions.
        Score alignment on a scale of 1-10 for each value. Explain reasoning.
        """
        
        analysis = self.llm.query(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1536,
            system_prompt=self.system_prompt
        )
        
        result = {
            "system_values": system_values,
            "action_sequence": action_sequence,
            "alignment_analysis": analysis,
            "timestamp": time.time()
        }
        
        # Extract alignment scores (simplified implementation)
        import re
        alignment_scores = {}
        
        for value in system_values:
            score_match = re.search(fr"{re.escape(value)}.+?(\d+)/10", analysis, re.DOTALL)
            if score_match:
                alignment_scores[value] = int(score_match.group(1))
                
        result["alignment_scores"] = alignment_scores
        
        # Calculate overall alignment
        if alignment_scores:
            result["overall_alignment"] = sum(alignment_scores.values()) / len(alignment_scores)
        
        # Store in memory if available
        if self.memory:
            memory_id = self.memory.store("value_alignment", result)
            result["memory_id"] = memory_id
            
        logger.info("Value alignment analysis completed")
        return result


class OrchestrationEngine:
    """
    Orchestrates cooperation between Raven and Seraph systems.
    """
    
    def __init__(self, raven: RavenIntelligence, seraph: SeraphIntelligence, memory_system: MemorySystem):
        """
        Initialize the orchestration engine.
        
        Args:
            raven: Raven intelligence system
            seraph: Seraph intelligence system
            memory_system: Memory system for storing and retrieving data
        """
        self.raven = raven
        self.seraph = seraph
        self.memory = memory_system
        logger.info("Initialized OrchestrationEngine")
    
    def process_pattern_with_ethical_review(self, pattern: str) -> Dict[str, Any]:
        """
        Process a pattern with Raven and then review implications with Seraph.
        
        Args:
            pattern: The pattern to process
            
        Returns:
            Combined results of pattern interpretation and ethical review
        """
        # First interpret the pattern with Raven
        interpretation_result = self.raven.interpret_pattern(pattern)
        interpretation = interpretation_result["interpretation"]
        
        # Define an action based on the interpretation
        action = f"Trigger memory recall based on: {interpretation}"
        
        # Evaluate the ethical implications with Seraph
        ethics_result = self.seraph.evaluate_ethics(
            action, 
            context={"pattern": pattern, "interpretation": interpretation}
        )
        
        # Store the combined process in memory
        combined_result = {
            "pattern": pattern,
            "interpretation_result": interpretation_result,
            "ethics_result": ethics_result,
            "timestamp": time.time()
        }
        
        memory_id = self.memory.store("orchestrated_process", combined_result)
        combined_result["memory_id"] = memory_id
        
        logger.info("Completed orchestrated process with ethical review")
        return combined_result
    
    def analyze_system_with_safety_checks(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a system with Raven and perform safety checks with Seraph.
        
        Args:
            system_data: Data describing the system to analyze
            
        Returns:
            Combined results of system analysis and safety evaluation
        """
        # First analyze the system with Raven
        analysis_result = self.raven.analyze_system_dynamics(system_data)
        
        # For each identified pattern, perform safety evaluation
        safety_evaluations = []
        for pattern in analysis_result.get("identified_patterns", []):
            safety_eval = self.seraph.evaluate_ethics(
                f"Implement system changes based on identified pattern: {pattern}",
                context={"system_data": system_data, "analysis": analysis_result["analysis"]}
            )
            safety_evaluations.append(safety_eval)
        
        # Summarize overall safety implications
        combined_result = {
            "system_data": system_data,
            "analysis_result": analysis_result,
            "safety_evaluations": safety_evaluations,
            "timestamp": time.time()
        }
        
        # Determine if overall system changes are safe to implement
        safe_patterns = []
        unsafe_patterns = []
        
        for i, eval_result in enumerate(safety_evaluations):
            pattern = analysis_result["identified_patterns"][i] if i < len(analysis_result["identified_patterns"]) else "Unknown pattern"
            
            # Simple heuristic: more benefits than concerns = safe
            if len(eval_result.get("benefits", [])) > len(eval_result.get("concerns", [])):
                safe_patterns.append(pattern)
            else:
                unsafe_patterns.append(pattern)
                
        combined_result["safe_patterns"] = safe_patterns
        combined_result["unsafe_patterns"] = unsafe_patterns
        combined_result["implementation_recommendation"] = "Proceed" if len(safe_patterns) > len(unsafe_patterns) else "Caution"
        
        # Store the combined result in memory
        memory_id = self.memory.store("safety_analysis", combined_result)
        combined_result["memory_id"] = memory_id
        
        logger.info("Completed system analysis with safety checks")
        return combined_result


# --- Demo Run ---
if __name__ == "__main__":
    # Initialize components
    llm = LLMAdapter()
    memory = MemorySystem()
    raven = RavenIntelligence(llm, memory)
    seraph = SeraphIntelligence(llm, memory)
    orchestrator = OrchestrationEngine(raven, seraph, memory)
    
    # Demo pattern
    pattern = "110010011001 - Synaptic burst encoding - Phase alignment: Positive"
    
    # Basic usage - interpret pattern
    interpretation_result = raven.interpret_pattern(pattern)
    print("[RAVEN INTERPRETATION]\n", interpretation_result["interpretation"])
    
    # Basic usage - evaluate ethics
    action = f"Trigger memory recall based on: {interpretation_result['interpretation']}"
    ethics_result = seraph.evaluate_ethics(action)
    print("\n[SERAPH ETHICAL EVALUATION]\n", ethics_result["evaluation"])
    
    # Advanced usage - orchestrated processing
    print("\n[ORCHESTRATED PROCESSING]")
    orchestrated_result = orchestrator.process_pattern_with_ethical_review(pattern)
    print("Process complete. Memory ID:", orchestrated_result["memory_id"])
    
    # Example of system analysis
    system_data = {
        "nodes": ["A", "B", "C", "D"],
        "connections": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
        "activation_levels": {"A": 0.85, "B": 0.42, "C": 0.91, "D": 0.36},
        "stability_index": 0.73
    }
    
    print("\n[SYSTEM ANALYSIS WITH SAFETY CHECKS]")
    analysis_result = orchestrator.analyze_system_with_safety_checks(system_data)
    print("Analysis complete.")
    print(f"Safe patterns: {len(analysis_result['safe_patterns'])}")
    print(f"Unsafe patterns: {len(analysis_result['unsafe_patterns'])}")
    print(f"Recommendation: {analysis_result['implementation_recommendation']}")
