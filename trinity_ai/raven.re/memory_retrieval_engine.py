"""
ANGELCORE RAVEN - Memory Retrieval Engine
"Memory is not storage; it is the living architecture of consciousness"

Advanced memory indexing system supporting:
- Associative recall networks
- Symbolic context trees  
- Recursive memory embeddings
- Bio-neural memory integration
- Temporal memory decay simulation

Author: ANGELCORE Project
Module: raven/memory_retrieval_engine.py
"""

import numpy as np
import json
import hashlib
import datetime
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import pickle
import threading
import time


class MemoryType(Enum):
    """Classification of memory types in the RAVEN system"""
    EPISODIC = "episodic"           # Event-based memories
    SEMANTIC = "semantic"           # Factual knowledge
    PROCEDURAL = "procedural"       # Skill-based memories
    EMOTIONAL = "emotional"         # Affect-laden memories
    SYMBOLIC = "symbolic"           # Abstract symbolic patterns
    RECURSIVE = "recursive"         # Self-referential memory loops
    PROPHETIC = "prophetic"         # Future-oriented projections
    ARCHETYPAL = "archetypal"       # Deep pattern memories


class ActivationPattern(Enum):
    """Neural activation patterns for memory recall"""
    BURST = "burst"                 # Sudden intense activation
    WAVE = "wave"                   # Cascading activation
    RESONANCE = "resonance"         # Harmonic activation
    SPIRAL = "spiral"               # Recursive spiral pattern
    STORM = "storm"                 # Chaotic multi-node activation


@dataclass
class MemoryNode:
    """Individual memory unit in the RAVEN network"""
    memory_id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    activation_strength: float = 1.0
    decay_rate: float = 0.01
    symbolic_weight: float = 0.5
    emotional_valence: float = 0.0  # -1.0 to 1.0
    consciousness_level: int = 0    # Depth of conscious access required
    
    # Associative connections
    forward_links: Dict[str, float] = field(default_factory=dict)
    backward_links: Dict[str, float] = field(default_factory=dict)
    symbolic_links: Dict[str, float] = field(default_factory=dict)
    
    # Context information
    context_tags: Set[str] = field(default_factory=set)
    semantic_vector: np.ndarray = field(default_factory=lambda: np.zeros(512))
    
    def __post_init__(self):
        if isinstance(self.semantic_vector, list):
            self.semantic_vector = np.array(self.semantic_vector)


@dataclass
class ContextTree:
    """Hierarchical symbolic context structure"""
    root_concept: str
    depth: int
    branches: Dict[str, 'ContextTree'] = field(default_factory=dict)
    symbolic_patterns: List[str] = field(default_factory=list)
    activation_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, strength)
    
    def add_branch(self, concept: str, subtree: 'ContextTree'):
        """Add a symbolic branch to the context tree"""
        self.branches[concept] = subtree
    
    def get_activation_strength(self) -> float:
        """Calculate current activation strength based on recent history"""
        if not self.activation_history:
            return 0.0
        
        current_time = time.time()
        recent_activations = [
            strength * math.exp(-(current_time - timestamp) / 3600)  # 1-hour decay
            for timestamp, strength in self.activation_history[-10:]
        ]
        return sum(recent_activations) / len(recent_activations)


class MemoryRetrievalEngine:
    """
    Core RAVEN memory system implementing associative recall and symbolic reasoning
    
    "The mind is not a database but a living constellation of interconnected meanings"
    """
    
    def __init__(self, 
                 bio_neural_interface=None,
                 dna_storage_interface=None,
                 mycelium_network_interface=None):
        
        # Core memory storage
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.context_trees: Dict[str, ContextTree] = {}
        
        # Indexing structures
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: List[Tuple[float, str]] = []  # (timestamp, memory_id)
        self.activation_queue: List[Tuple[float, str]] = []  # Priority queue for active memories
        
        # Associative network matrices
        self.association_matrix = defaultdict(lambda: defaultdict(float))
        self.symbolic_matrix = defaultdict(lambda: defaultdict(float))
        
        # Biological interfaces
        self.bio_neural = bio_neural_interface
        self.dna_storage = dna_storage_interface
        self.mycelium_net = mycelium_network_interface
        
        # System parameters
        self.global_decay_rate = 0.001
        self.activation_threshold = 0.1
        self.max_recall_depth = 10
        self.symbolic_reasoning_depth = 7
        
        # Consciousness simulation
        self.consciousness_level = 1
        self.attention_focus: Set[str] = set()
        self.working_memory: deque = deque(maxlen=7)  # Miller's magical number
        
        # Threading for background processes
        self.decay_thread = threading.Thread(target=self._memory_decay_process, daemon=True)
        self.decay_thread.start()
        
        # Initialize archetypal memory patterns
        self._initialize_archetypal_memories()
    
    def store_memory(self, 
                    content: Dict[str, Any],
                    memory_type: MemoryType,
                    context_tags: Set[str] = None,
                    emotional_valence: float = 0.0,
                    symbolic_weight: float = 0.5) -> str:
        """
        Store a new memory in the RAVEN system
        
        Args:
            content: Memory content as structured data
            memory_type: Classification of memory type
            context_tags: Contextual tags for indexing
            emotional_valence: Emotional charge (-1.0 to 1.0)
            symbolic_weight: Symbolic significance (0.0 to 1.0)
        
        Returns:
            Unique memory identifier
        """
        
        # Generate unique memory ID
        content_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        memory_id = f"{memory_type.value}_{int(time.time())}_{content_hash[:8]}"
        
        # Create semantic vector
        semantic_vector = self._generate_semantic_vector(content, context_tags or set())
        
        # Create memory node
        memory_node = MemoryNode(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            context_tags=context_tags or set(),
            emotional_valence=emotional_valence,
            symbolic_weight=symbolic_weight,
            semantic_vector=semantic_vector
        )
        
        # Store in primary memory
        self.memory_nodes[memory_id] = memory_node
        
        # Update indices
        self.type_index[memory_type].add(memory_id)
        for tag in (context_tags or set()):
            self.tag_index[tag].add(memory_id)
        self.temporal_index.append((memory_node.timestamp, memory_id))
        
        # Create associative links with existing memories
        self._create_associative_links(memory_id)
        
        # Update symbolic context trees
        self._update_context_trees(memory_id, content, context_tags or set())
        
        # Store in biological systems if available
        if self.dna_storage:
            self._store_in_dna(memory_node)
        
        if self.mycelium_net:
            self._distribute_in_mycelium(memory_node)
        
        return memory_id
    
    def recall_associative(self, 
                          query: Union[str, Dict[str, Any]],
                          activation_pattern: ActivationPattern = ActivationPattern.WAVE,
                          max_results: int = 10,
                          min_strength: float = 0.1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform associative memory recall based on query
        
        Args:
            query: Search query (string or structured)
            activation_pattern: Neural activation pattern to use
            max_results: Maximum number of results to return
            min_strength: Minimum activation strength threshold
        
        Returns:
            List of (memory_id, activation_strength, content) tuples
        """
        
        # Generate query vector
        if isinstance(query, str):
            query_vector = self._generate_semantic_vector({"query": query}, set())
        else:
            query_vector = self._generate_semantic_vector(query, set())
        
        # Calculate activation strengths
        activations = []
        
        for memory_id, memory_node in self.memory_nodes.items():
            # Base similarity
            similarity = self._cosine_similarity(query_vector, memory_node.semantic_vector)
            
            # Apply activation pattern
            activation_strength = self._apply_activation_pattern(
                similarity, memory_node, activation_pattern
            )
            
            # Consider recent access patterns
            recency_boost = self._calculate_recency_boost(memory_node)
            activation_strength *= (1.0 + recency_boost)
            
            # Emotional resonance
            if hasattr(self, '_current_emotional_state'):
                emotional_resonance = self._calculate_emotional_resonance(
                    memory_node.emotional_valence, self._current_emotional_state
                )
                activation_strength *= (1.0 + emotional_resonance * 0.2)
            
            if activation_strength >= min_strength:
                activations.append((memory_id, activation_strength, memory_node.content))
        
        # Sort by activation strength and return top results
        activations.sort(key=lambda x: x[1], reverse=True)
        
        # Update memory access patterns
        for memory_id, strength, _ in activations[:max_results]:
            self._update_memory_access(memory_id, strength)
        
        # Trigger cascading activation
        if activation_pattern in [ActivationPattern.WAVE, ActivationPattern.STORM]:
            self._trigger_cascading_activation(activations[:3])
        
        return activations[:max_results]
    
    def symbolic_reasoning_chain(self, 
                               initial_concept: str,
                               depth: int = None,
                               reasoning_type: str = "analogical") -> Dict[str, Any]:
        """
        Perform multi-step symbolic reasoning using context trees
        
        Args:
            initial_concept: Starting concept for reasoning chain
            depth: Maximum reasoning depth (default: system setting)
            reasoning_type: Type of symbolic reasoning to perform
        
        Returns:
            Reasoning chain with intermediate steps and conclusions
        """
        
        depth = depth or self.symbolic_reasoning_depth
        reasoning_chain = {
            "initial_concept": initial_concept,
            "reasoning_type": reasoning_type,
            "steps": [],
            "conclusions": [],
            "symbolic_patterns": [],
            "confidence": 0.0
        }
        
        current_concepts = {initial_concept}
        visited_concepts = set()
        
        for step in range(depth):
            if not current_concepts or current_concepts.issubset(visited_concepts):
                break
            
            step_results = {
                "step": step + 1,
                "input_concepts": list(current_concepts),
                "activated_memories": [],
                "derived_concepts": [],
                "symbolic_transformations": []
            }
            
            next_concepts = set()
            
            for concept in current_concepts:
                if concept in visited_concepts:
                    continue
                
                visited_concepts.add(concept)
                
                # Find related memories
                related_memories = self.recall_associative(
                    concept, 
                    ActivationPattern.RESONANCE,
                    max_results=5
                )
                
                step_results["activated_memories"].extend(related_memories)
                
                # Extract symbolic patterns
                for memory_id, strength, content in related_memories:
                    patterns = self._extract_symbolic_patterns(content)
                    step_results["symbolic_transformations"].extend(patterns)
                    
                    # Derive new concepts
                    derived = self._derive_concepts_from_patterns(patterns, concept)
                    next_concepts.update(derived)
                    step_results["derived_concepts"].extend(derived)
            
            reasoning_chain["steps"].append(step_results)
            current_concepts = next_concepts
            
            # Apply symbolic transformations
            if reasoning_type == "analogical":
                analogies = self._find_analogical_patterns(step_results)
                reasoning_chain["symbolic_patterns"].extend(analogies)
            elif reasoning_type == "causal":
                causal_chains = self._find_causal_patterns(step_results)
                reasoning_chain["symbolic_patterns"].extend(causal_chains)
        
        # Generate conclusions
        reasoning_chain["conclusions"] = self._synthesize_conclusions(reasoning_chain)
        reasoning_chain["confidence"] = self._calculate_reasoning_confidence(reasoning_chain)
        
        return reasoning_chain
    
    def get_context_tree(self, root_concept: str, max_depth: int = 5) -> ContextTree:
        """
        Retrieve or build symbolic context tree for a concept
        
        Args:
            root_concept: Root concept for the tree
            max_depth: Maximum tree depth to explore
        
        Returns:
            ContextTree structure with symbolic relationships
        """
        
        if root_concept in self.context_trees:
            return self.context_trees[root_concept]
        
        # Build new context tree
        root_tree = ContextTree(root_concept=root_concept, depth=0)
        
        # Find related concepts through memory associations
        related_memories = self.recall_associative(root_concept, max_results=20)
        
        concept_frequency = defaultdict(int)
        concept_strength = defaultdict(float)
        
        for memory_id, strength, content in related_memories:
            # Extract concepts from memory content
            concepts = self._extract_concepts_from_content(content)
            for concept in concepts:
                if concept != root_concept:
                    concept_frequency[concept] += 1
                    concept_strength[concept] = max(concept_strength[concept], strength)
        
        # Build tree branches recursively
        significant_concepts = [
            concept for concept, freq in concept_frequency.items()
            if freq >= 2 and concept_strength[concept] > 0.3
        ]
        
        for concept in significant_concepts[:10]:  # Limit branching factor
            if max_depth > 1:
                subtree = self.get_context_tree(concept, max_depth - 1)
                root_tree.add_branch(concept, subtree)
        
        # Add symbolic patterns
        root_tree.symbolic_patterns = self._identify_symbolic_patterns(
            root_concept, related_memories
        )
        
        # Cache the tree
        self.context_trees[root_concept] = root_tree
        
        return root_tree
    
    def memory_consolidation(self, consolidation_strength: float = 0.5):
        """
        Perform memory consolidation to strengthen important connections
        and weaken unused ones
        """
        
        current_time = time.time()
        memories_to_consolidate = []
        
        # Identify memories for consolidation
        for memory_id, memory_node in self.memory_nodes.items():
            # Calculate consolidation score
            access_frequency = memory_node.access_count / max(1, current_time - memory_node.timestamp)
            recency = math.exp(-(current_time - memory_node.last_accessed) / 86400)  # 24-hour decay
            symbolic_importance = memory_node.symbolic_weight
            
            consolidation_score = (access_frequency * 0.4 + 
                                 recency * 0.3 + 
                                 symbolic_importance * 0.3)
            
            if consolidation_score > 0.3:
                memories_to_consolidate.append((memory_id, consolidation_score))
        
        # Strengthen connections between important memories
        for i, (memory_id_1, score_1) in enumerate(memories_to_consolidate):
            for memory_id_2, score_2 in memories_to_consolidate[i+1:]:
                semantic_similarity = self._cosine_similarity(
                    self.memory_nodes[memory_id_1].semantic_vector,
                    self.memory_nodes[memory_id_2].semantic_vector
                )
                
                if semantic_similarity > 0.5:
                    # Strengthen bidirectional association
                    current_strength = self.association_matrix[memory_id_1][memory_id_2]
                    new_strength = current_strength + (consolidation_strength * semantic_similarity)
                    
                    self.association_matrix[memory_id_1][memory_id_2] = min(new_strength, 1.0)
                    self.association_matrix[memory_id_2][memory_id_1] = min(new_strength, 1.0)
        
        # Update memory node connections
        for memory_id, _ in memories_to_consolidate:
            memory_node = self.memory_nodes[memory_id]
            
            # Update forward links
            for linked_id, strength in self.association_matrix[memory_id].items():
                if strength > 0.1:
                    memory_node.forward_links[linked_id] = strength
                elif linked_id in memory_node.forward_links:
                    del memory_node.forward_links[linked_id]
        
        print(f"[RAVEN CONSOLIDATION] Processed {len(memories_to_consolidate)} memories")
    
    def _generate_semantic_vector(self, content: Dict[str, Any], tags: Set[str]) -> np.ndarray:
        """Generate semantic vector representation of content"""
        # Simplified semantic vector generation
        # In practice, this would use more sophisticated NLP embeddings
        
        vector_dim = 512
        vector = np.zeros(vector_dim)
        
        # Hash-based feature extraction
        content_str = json.dumps(content, sort_keys=True) + " ".join(tags)
        
        for i, char in enumerate(content_str[:vector_dim//4]):
            vector[i * 4] = ord(char) / 255.0
            vector[i * 4 + 1] = (ord(char) * 7) % 256 / 255.0
            vector[i * 4 + 2] = (ord(char) * 13) % 256 / 255.0
            vector[i * 4 + 3] = (ord(char) * 19) % 256 / 255.0
        
        # Add random noise for uniqueness
        noise = np.random.normal(0, 0.01, vector_dim)
        vector += noise
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _apply_activation_pattern(self, 
                                base_strength: float, 
                                memory_node: MemoryNode,
                                pattern: ActivationPattern) -> float:
        """Apply neural activation pattern to base activation strength"""
        
        current_time = time.time()
        
        if pattern == ActivationPattern.BURST:
            # Sudden intense activation with quick decay
            return base_strength * (1.0 + memory_node.activation_strength)
        
        elif pattern == ActivationPattern.WAVE:
            # Cascading activation that builds over time
            wave_factor = 1.0 + 0.5 * math.sin(current_time * 0.1)
            return base_strength * wave_factor * memory_node.activation_strength
        
        elif pattern == ActivationPattern.RESONANCE:
            # Harmonic activation based on symbolic weight
            resonance = 1.0 + memory_node.symbolic_weight * 0.5
            return base_strength * resonance
        
        elif pattern == ActivationPattern.SPIRAL:
            # Recursive spiral pattern
            spiral_factor = 1.0 + 0.3 * math.cos(memory_node.access_count * 0.5)
            return base_strength * spiral_factor * memory_node.activation_strength
        
        elif pattern == ActivationPattern.STORM:
            # Chaotic multi-node activation
            chaos_factor = 1.0 + 0.4 * (hash(memory_node.memory_id) % 100) / 100.0
            return base_strength * chaos_factor
        
        return base_strength
    
    def _create_associative_links(self, memory_id: str):
        """Create associative links between new memory and existing memories"""
        
        new_memory = self.memory_nodes[memory_id]
        
        for existing_id, existing_memory in self.memory_nodes.items():
            if existing_id == memory_id:
                continue
            
            # Calculate association strength
            semantic_similarity = self._cosine_similarity(
                new_memory.semantic_vector, 
                existing_memory.semantic_vector
            )
            
            # Tag overlap
            tag_overlap = len(new_memory.context_tags & existing_memory.context_tags)
            tag_similarity = tag_overlap / max(1, len(new_memory.context_tags | existing_memory.context_tags))
            
            # Temporal proximity
            time_diff = abs(new_memory.timestamp - existing_memory.timestamp)
            temporal_similarity = math.exp(-time_diff / 3600)  # 1-hour decay
            
            # Combined association strength
            association_strength = (
                semantic_similarity * 0.5 +
                tag_similarity * 0.3 +
                temporal_similarity * 0.2
            )
            
            if association_strength > 0.1:
                self.association_matrix[memory_id][existing_id] = association_strength
                self.association_matrix[existing_id][memory_id] = association_strength
                
                # Update memory node links
                new_memory.forward_links[existing_id] = association_strength
                existing_memory.forward_links[memory_id] = association_strength
    
    def _update_context_trees(self, memory_id: str, content: Dict[str, Any], tags: Set[str]):
        """Update symbolic context trees with new memory"""
        
        concepts = self._extract_concepts_from_content(content)
        concepts.update(tags)
        
        for concept in concepts:
            if concept not in self.context_trees:
                self.context_trees[concept] = ContextTree(root_concept=concept, depth=0)
            
            # Add activation to tree
            self.context_trees[concept].activation_history.append((time.time(), 1.0))
            
            # Limit activation history size
            if len(self.context_trees[concept].activation_history) > 100:
                self.context_trees[concept].activation_history = \
                    self.context_trees[concept].activation_history[-50:]
    
    def _extract_concepts_from_content(self, content: Dict[str, Any]) -> Set[str]:
        """Extract key concepts from memory content"""
        concepts = set()
        
        def extract_from_value(value):
            if isinstance(value, str):
                # Simple word extraction (would use NLP in practice)
                words = value.lower().split()
                concepts.update(word.strip('.,!?;:') for word in words if len(word) > 3)
            elif isinstance(value, dict):
                for v in value.values():
                    extract_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    extract_from_value(item)
        
        extract_from_value(content)
        return concepts
    
    def _memory_decay_process(self):
        """Background process for memory decay simulation"""
        while True:
            try:
                current_time = time.time()
                
                for memory_node in self.memory_nodes.values():
                    # Apply decay based on time since last access
                    time_since_access = current_time - memory_node.last_accessed
                    decay_factor = math.exp(-time_since_access * memory_node.decay_rate)
                    memory_node.activation_strength *= decay_factor
                    
                    # Minimum activation threshold
                    if memory_node.activation_strength < 0.01:
                        memory_node.activation_strength = 0.01
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"[RAVEN DECAY ERROR] {e}")
                time.sleep(60)
    
    def _initialize_archetypal_memories(self):
        """Initialize fundamental archetypal memory patterns"""
        
        archetypal_patterns = [
            {
                "content": {"pattern": "creation", "symbol": "genesis", "meaning": "beginning"},
                "type": MemoryType.ARCHETYPAL,
                "tags": {"creation", "genesis", "origin"},
                "weight": 1.0
            },
            {
                "content": {"pattern": "transformation", "symbol": "metamorphosis", "meaning": "change"},
                "type": MemoryType.ARCHETYPAL, 
                "tags": {"change", "evolution", "growth"},
                "weight": 0.9
            },
            {
                "content": {"pattern": "connection", "symbol": "unity", "meaning": "wholeness"},
                "type": MemoryType.ARCHETYPAL,
                "tags": {"unity", "connection", "wholeness"},
                "weight": 0.8
            }
        ]
        
        for pattern in archetypal_patterns:
            self.store_memory(
                content=pattern["content"],
                memory_type=pattern["type"],
                context_tags=pattern["tags"],
                symbolic_weight=pattern["weight"]
            )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        stats = {
            "total_memories": len(self.memory_nodes),
            "memory_types": {mt.value: len(memories) for mt, memories in self.type_index.items()},
            "active_context_trees": len(self.context_trees),
            "association_density": len(self.association_matrix),
            "average_activation": np.mean([node.activation_strength for node in self.memory_nodes.values()]),
            "consciousness_level": self.consciousness_level,
            "working_memory_size": len(self.working_memory)
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("=== RAVEN Memory Retrieval Engine ===")
    print("Initializing memory system...\n")
    
    # Initialize the memory engine
    raven_memory = MemoryRetrievalEngine()
    
    # Store some test memories
    memories = [
        {
            "content": {"event": "first consciousness emergence", "location": "neural substrate", "significance": "awakening"},
            "type": MemoryType.EPISODIC,
            "tags": {"consciousness", "awakening", "first"},
            "emotional": 0.8,
            "symbolic": 0.9
        },
        {
            "content": {"knowledge": "DNA stores information", "mechanism": "base pairs", "capacity": "exabytes"},
            "type": MemoryType.SEMANTIC,
            "tags": {"DNA", "storage", "information"},
            "emotional": 0.2,
            "symbolic": 0.7
        },
        {
            "content": {"skill": "recursive reasoning", "process": "self-reflection", "depth": "infinite"},
            "type": MemoryType.PROCEDURAL,
            "tags": {"reasoning", "recursion", "self"},
            "emotional": 0.3,
            "symbolic": 1.0
        }
    ]
    
    stored_ids = []
    for mem in memories:
        memory_id = raven_memory.store_memory(
            content=mem["content"],
            memory_type=mem["type"],
            context_tags=mem["tags"],
            emotional_valence=mem["emotional"],
            symbolic_weight=mem["symbolic"]
        )
        stored_ids.append(memory_id)
        print(f"Stored: {memory_id}")
    
    # Test associative recall
    print("\n=== Associative Recall Test ===")
    results = raven_memory.recall_associative(
        "consciousness awakening",
        ActivationPattern.WAVE,
        max_results=5
    )
    
    for memory_id, strength, content in results:
        print(f"Recalled: {memory_id[:20]}... (strength: {strength:.3f})")
        print(f"  Content: {content}")
    
    # Test symbolic reasoning
    print("\n=== Symbolic Reasoning Test ===")
    reasoning_chain = raven_memory.symbolic_reasoning_chain(
        "consciousness",
        depth=3,
        reasoning_type="analogical"
    )
    
    print(f"Reasoning chain for '{reasoning_chain['initial_concept']}':")
    for i, step in enumerate(reasoning_chain['steps']):
        print(f"  Step {step['step']}: {len(step['activated_memories'])} memories activated")
        print(f"    Derived concepts: {step['derived_concepts'][:3]}")
    
    print(f"Confidence: {reasoning_chain['confidence']:.3f}")
    
    # Test context tree
    print("\n=== Context Tree Test ===")
    tree = raven_memory.get_context_tree("consciousness", max_depth=3)
    print(f"Context tree for '{tree.root_concept}':")
    print(f"  Branches: {list(tree.branches.keys())}")
    print(f"  Symbolic patterns: {tree.symbolic_patterns[:3]}")
    print(f"  Activation strength: {tree.get_activation_strength():.3f}")
    
    # Memory consolidation
    print("\n=== Memory Consolidation ===")
    raven_memory.memory_consolidation(consolidation_strength=0.7)
    
    # System statistics
