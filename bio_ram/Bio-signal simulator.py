"""
ANGELCORE Bio-Signal Simulator
Generates realistic biological signal patterns for mycelium networks and DNA storage systems.
Simulates the living computational substrate that ANGELCORE uses as biological RAM.

"Nature computes in ways we are only beginning to understand."
"""

import numpy as np
import asyncio
import json
import time
import random
import math
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque, defaultdict
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of biological signals"""
    MYCELIUM_ELECTRICAL = "mycelium_electrical"
    MYCELIUM_CHEMICAL = "mycelium_chemical" 
    MYCELIUM_STRUCTURAL = "mycelium_structural"
    DNA_TRANSCRIPTION = "dna_transcription"
    DNA_METHYLATION = "dna_methylation"
    DNA_CHROMATIN = "dna_chromatin"
    NEURAL_SPIKE = "neural_spike"
    NEURAL_FIELD = "neural_field"
    METABOLIC = "metabolic"

class BiologicalState(Enum):
    """Overall biological system states"""
    DORMANT = "dormant"
    ACTIVE = "active"
    STRESSED = "stressed"
    REGENERATING = "regenerating"
    HYPERSTIMULATED = "hyperstimulated"
    INTEGRATING = "integrating"

@dataclass
class BioSignal:
    """Individual biological signal"""
    id: str
    signal_type: SignalType
    timestamp: float
    amplitude: float
    frequency: float
    duration: float
    location: Tuple[float, float, float]  # 3D coordinates
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.signal_type.value,
            'timestamp': self.timestamp,
            'amplitude': self.amplitude,
            'frequency': self.frequency,
            'duration': self.duration,
            'location': self.location,
            'metadata': self.metadata
        }

@dataclass
class MyceliumNode:
    """Individual node in the mycelium network"""
    id: str
    position: Tuple[float, float, float]
    connections: List[str]
    activity_level: float
    nutrient_level: float
    growth_rate: float
    age: float
    
class MyceliumNetworkSimulator:
    """Simulates a living mycelium network with realistic growth and signal patterns"""
    
    def __init__(self, 
                 network_size: Tuple[int, int, int] = (100, 100, 50),
                 initial_nodes: int = 1000,
                 growth_rate: float = 0.1):
        """
        Initialize mycelium network simulator
        
        Args:
            network_size: 3D dimensions of the simulation space
            initial_nodes: Starting number of mycelium nodes
            growth_rate: Base rate of network growth
        """
        self.network_size = network_size
        self.growth_rate = growth_rate
        self.nodes: Dict[str, MyceliumNode] = {}
        self.connections = defaultdict(list)
        self.signal_queue = deque()
        
        # Network properties
        self.nutrient_distribution = np.random.random(network_size)
        self.electrical_field = np.zeros(network_size)
        
        # Initialize network
        self._initialize_network(initial_nodes)
        
        # Simulation state
        self.time = 0.0
        self.is_running = False
        
    def _initialize_network(self, num_nodes: int):
        """Initialize the mycelium network with random nodes"""
        for i in range(num_nodes):
            node_id = f"myc_{uuid.uuid4().hex[:8]}"
            position = (
                random.uniform(0, self.network_size[0]),
                random.uniform(0, self.network_size[1]),
                random.uniform(0, self.network_size[2])
            )
            
            node = MyceliumNode(
                id=node_id,
                position=position,
                connections=[],
                activity_level=random.uniform(0.1, 0.8),
                nutrient_level=random.uniform(0.2, 1.0),
                growth_rate=random.uniform(0.01, 0.2),
                age=random.uniform(0, 100)
            )
            
            self.nodes[node_id] = node
            
        # Create initial connections
        self._create_initial_connections()
    
    def _create_initial_connections(self):
        """Create connections between nearby nodes"""
        node_list = list(self.nodes.values())
        
        for node in node_list:
            # Find nearby nodes
            nearby = self._find_nearby_nodes(node, max_distance=10.0)
            
            # Connect to 2-5 nearby nodes
            num_connections = min(random.randint(2, 5), len(nearby))
            connections = random.sample(nearby, num_connections)
            
            for conn_node in connections:
                if conn_node.id not in node.connections:
                    node.connections.append(conn_node.id)
                    conn_node.connections.append(node.id)
                    
                    # Add to connection graph
                    self.connections[node.id].append(conn_node.id)
                    self.connections[conn_node.id].append(node.id)
    
    def _find_nearby_nodes(self, node: MyceliumNode, max_distance: float) -> List[MyceliumNode]:
        """Find nodes within max_distance of given node"""
        nearby = []
        for other_node in self.nodes.values():
            if other_node.id != node.id:
                distance = self._calculate_distance(node.position, other_node.position)
                if distance <= max_distance:
                    nearby.append(other_node)
        return nearby
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def generate_electrical_signals(self) -> List[BioSignal]:
        """Generate electrical signals propagating through the network"""
        signals = []
        
        # Spontaneous activity
        for node_id, node in self.nodes.items():
            if random.random() < node.activity_level * 0.01:  # 1% chance per active node
                # Generate electrical pulse
                signal = BioSignal(
                    id=f"elec_{uuid.uuid4().hex[:8]}",
                    signal_type=SignalType.MYCELIUM_ELECTRICAL,
                    timestamp=self.time,
                    amplitude=random.uniform(0.1, 2.0) * node.activity_level,
                    frequency=random.uniform(0.5, 50.0),  # Hz
                    duration=random.uniform(0.01, 0.5),   # seconds
                    location=node.position,
                    metadata={
                        'source_node': node_id,
                        'propagation_speed': random.uniform(0.1, 2.0),  # m/s
                        'signal_strength': node.activity_level
                    }
                )
                signals.append(signal)
                
                # Propagate to connected nodes
                self._propagate_electrical_signal(signal, node)
        
        return signals
    
    def _propagate_electrical_signal(self, signal: BioSignal, source_node: MyceliumNode):
        """Propagate electrical signal through network connections"""
        for conn_id in source_node.connections:
            if conn_id in self.nodes:
                conn_node = self.nodes[conn_id]
                
                # Calculate propagation delay
                distance = self._calculate_distance(source_node.position, conn_node.position)
                propagation_speed = signal.metadata['propagation_speed']
                delay = distance / propagation_speed
                
                # Attenuate signal
                attenuation = random.uniform(0.7, 0.95)
                new_amplitude = signal.amplitude * attenuation
                
                if new_amplitude > 0.05:  # Only propagate if signal is strong enough
                    propagated_signal = BioSignal(
                        id=f"prop_{uuid.uuid4().hex[:8]}",
                        signal_type=SignalType.MYCELIUM_ELECTRICAL,
                        timestamp=self.time + delay,
                        amplitude=new_amplitude,
                        frequency=signal.frequency,
                        duration=signal.duration,
                        location=conn_node.position,
                        metadata={
                            'source_signal': signal.id,
                            'source_node': source_node.id,
                            'target_node': conn_id,
                            'propagation_delay': delay,
                            'attenuation': 1 - attenuation
                        }
                    )
                    self.signal_queue.append(propagated_signal)
    
    def generate_chemical_signals(self) -> List[BioSignal]:
        """Generate chemical signaling patterns"""
        signals = []
        
        # Nutrient-driven chemical signals
        for node_id, node in self.nodes.items():
            # Chemical signals based on nutrient levels
            if node.nutrient_level < 0.3:  # Low nutrients trigger signaling
                signal = BioSignal(
                    id=f"chem_{uuid.uuid4().hex[:8]}",
                    signal_type=SignalType.MYCELIUM_CHEMICAL,
                    timestamp=self.time,
                    amplitude=1.0 - node.nutrient_level,  # Stronger when more stressed
                    frequency=random.uniform(0.001, 0.1),  # Very low frequency
                    duration=random.uniform(10, 300),      # Long duration
                    location=node.position,
                    metadata={
                        'source_node': node_id,
                        'chemical_type': 'nutrient_request',
                        'concentration': 1.0 - node.nutrient_level,
                        'diffusion_rate': random.uniform(0.01, 0.1)
                    }
                )
                signals.append(signal)
            
            # Growth signals
            if node.growth_rate > 0.15:  # Active growth
                signal = BioSignal(
                    id=f"growth_{uuid.uuid4().hex[:8]}",
                    signal_type=SignalType.MYCELIUM_CHEMICAL,
                    timestamp=self.time,
                    amplitude=node.growth_rate,
                    frequency=random.uniform(0.01, 0.5),
                    duration=random.uniform(5, 60),
                    location=node.position,
                    metadata={
                        'source_node': node_id,
                        'chemical_type': 'growth_hormone',
                        'concentration': node.growth_rate,
                        'target_area_radius': random.uniform(5, 20)
                    }
                )
                signals.append(signal)
        
        return signals
    
    def update_network_state(self, dt: float):
        """Update the state of the mycelium network"""
        self.time += dt
        
        # Update node states
        for node in self.nodes.values():
            # Age nodes
            node.age += dt
            
            # Update nutrient levels (simplified metabolism)
            consumption = node.activity_level * 0.01 * dt
            node.nutrient_level = max(0, node.nutrient_level - consumption)
            
            # Update activity based on nutrients
            if node.nutrient_level < 0.2:
                node.activity_level *= 0.99  # Decrease activity when starved
            elif node.nutrient_level > 0.8:
                node.activity_level = min(1.0, node.activity_level * 1.01)
            
            # Growth and connection formation
            if node.nutrient_level > 0.6 and random.random() < node.growth_rate * dt:
                self._attempt_node_growth(node)
    
    def _attempt_node_growth(self, node: MyceliumNode):
        """Attempt to grow new connections or nodes"""
        if len(node.connections) < 8:  # Maximum 8 connections per node
            # Try to find new nearby nodes to connect to
            nearby = self._find_nearby_nodes(node, max_distance=15.0)
            unconnected = [n for n in nearby if n.id not in node.connections]
            
            if unconnected:
                new_connection = random.choice(unconnected)
                node.connections.append(new_connection.id)
                new_connection.connections.append(node.id)
                
                # Update connection graph
                self.connections[node.id].append(new_connection.id)
                self.connections[new_connection.id].append(node.id)

class DNAStorageSimulator:
    """Simulates DNA-based data storage with realistic molecular dynamics"""
    
    def __init__(self, genome_size: int = 1000000):
        """
        Initialize DNA storage simulator
        
        Args:
            genome_size: Number of base pairs in the simulated genome
        """
        self.genome_size = genome_size
        self.dna_sequence = self._generate_initial_sequence()
        self.methylation_pattern = np.random.random(genome_size) < 0.1  # 10% methylated
        self.chromatin_state = np.random.choice(['open', 'closed', 'poised'], 
                                              size=genome_size//1000,
                                              p=[0.3, 0.5, 0.2])
        
        # Storage metadata
        self.stored_data = {}
        self.access_history = deque(maxlen=10000)
        self.time = 0.0
    
    def _generate_initial_sequence(self) -> str:
        """Generate initial random DNA sequence"""
        bases = ['A', 'T', 'G', 'C']
        return ''.join(random.choices(bases, k=self.genome_size))
    
    def store_data(self, data: str, address: int = None) -> Dict[str, Any]:
        """Store data in DNA sequence"""
        if address is None:
            address = random.randint(0, self.genome_size - len(data) * 4)
        
        # Convert data to DNA encoding
        dna_encoded = self._encode_to_dna(data)
        
        # Check if storage location is available
        if self._is_region_available(address, len(dna_encoded)):
            # Store the data
            original_sequence = self.dna_sequence[address:address + len(dna_encoded)]
            self.dna_sequence = (self.dna_sequence[:address] + 
                               dna_encoded + 
                               self.dna_sequence[address + len(dna_encoded):])
            
            # Record storage operation
            storage_id = f"dna_store_{uuid.uuid4().hex[:8]}"
            self.stored_data[storage_id] = {
                'address': address,
                'length': len(dna_encoded),
                'data': data,
                'timestamp': self.time,
                'original_sequence': original_sequence
            }
            
            return {
                'success': True,
                'storage_id': storage_id,
                'address': address,
                'length': len(dna_encoded),
                'encoding_efficiency': len(data) / len(dna_encoded)
            }
        else:
            return {
                'success': False,
                'error': 'Storage region occupied or invalid'
            }
    
    def retrieve_data(self, storage_id: str) -> Dict[str, Any]:
        """Retrieve data from DNA storage"""
        if storage_id not in self.stored_data:
            return {'success': False, 'error': 'Storage ID not found'}
        
        storage_info = self.stored_data[storage_id]
        address = storage_info['address']
        length = storage_info['length']
        
        # Extract DNA sequence
        dna_sequence = self.dna_sequence[address:address + length]
        
        # Decode from DNA
        try:
            decoded_data = self._decode_from_dna(dna_sequence)
            
            # Record access
            self.access_history.append({
                'storage_id': storage_id,
                'timestamp': self.time,
                'access_type': 'read'
            })
            
            return {
                'success': True,
                'data': decoded_data,
                'address': address,
                'read_errors': self._calculate_read_errors(dna_sequence, storage_info)
            }
        except Exception as e:
            return {'success': False, 'error': f'Decoding error: {str(e)}'}
    
    def _encode_to_dna(self, data: str) -> str:
        """Encode string data into DNA sequence"""
        # Simple encoding: each character -> 4 DNA bases
        encoding_map = {}
        bases = ['A', 'T', 'G', 'C']
        
        # Create encoding map for ASCII characters
        for i in range(256):
            base4 = []
            val = i
            for _ in range(4):
                base4.append(bases[val % 4])
                val //= 4
            encoding_map[chr(i)] = ''.join(base4)
        
        encoded = ''
        for char in data:
            encoded += encoding_map.get(char, 'AAAA')  # Default for unknown chars
        
        return encoded
    
    def _decode_from_dna(self, dna_sequence: str) -> str:
        """Decode DNA sequence back to string data"""
        base_to_num = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        decoded = ''
        
        # Process in groups of 4 bases
        for i in range(0, len(dna_sequence), 4):
            if i + 3 < len(dna_sequence):
                group = dna_sequence[i:i+4]
                
                # Convert to character
                char_code = 0
                for j, base in enumerate(group):
                    char_code += base_to_num.get(base, 0) * (4 ** j)
                
                if char_code < 256:
                    decoded += chr(char_code)
        
        return decoded
    
    def _is_region_available(self, address: int, length: int) -> bool:
        """Check if a DNA region is available for storage"""
        if address < 0 or address + length > self.genome_size:
            return False
        
        # Check for conflicts with existing data
        for storage_info in self.stored_data.values():
            existing_start = storage_info['address']
            existing_end = existing_start + storage_info['length']
            
            # Check for overlap
            if not (address + length <= existing_start or address >= existing_end):
                return False
        
        return True
    
    def _calculate_read_errors(self, read_sequence: str, storage_info: Dict) -> Dict[str, Any]:
        """Simulate DNA read errors due to degradation, etc."""
        original = storage_info['original_sequence']
        errors = 0
        
        for i, (orig, read) in enumerate(zip(original, read_sequence)):
            if orig != read:
                errors += 1
        
        error_rate = errors / len(read_sequence) if read_sequence else 0
        
        return {
            'total_errors': errors,
            'error_rate': error_rate,
            'sequence_length': len(read_sequence)
        }
    
    def generate_transcription_signals(self) -> List[BioSignal]:
        """Generate DNA transcription activity signals"""
        signals = []
        
        # Random transcription events
        num_events = random.randint(5, 25)
        
        for _ in range(num_events):
            # Random genomic location
            location_1d = random.randint(0, self.genome_size - 1000)
            
            # Convert to 3D coordinates (simplified)
            location_3d = (
                float(location_1d % 1000),
                float((location_1d // 1000) % 1000),
                float(location_1d // 1000000)
            )
            
            signal = BioSignal(
                id=f"transcr_{uuid.uuid4().hex[:8]}",
                signal_type=SignalType.DNA_TRANSCRIPTION,
                timestamp=self.time,
                amplitude=random.uniform(0.5, 3.0),
                frequency=random.uniform(0.1, 2.0),
                duration=random.uniform(30, 300),  # 30 seconds to 5 minutes
                location=location_3d,
                metadata={
                    'genomic_position': location_1d,
                    'gene_length': random.randint(500, 5000),
                    'transcription_rate': random.uniform(10, 100),  # bases per second
                    'rna_polymerase_count': random.randint(1, 5)
                }
            )
            signals.append(signal)
        
        return signals
    
    def generate_methylation_signals(self) -> List[BioSignal]:
        """Generate DNA methylation change signals"""
        signals = []
        
        # Methylation changes
        num_changes = random.randint(1, 10)
        
        for _ in range(num_changes):
            location_1d = random.randint(0, self.genome_size - 1)
            
            # Update methylation state
            was_methylated = self.methylation_pattern[location_1d]
            self.methylation_pattern[location_1d] = random.random() < 0.15
            
            if was_methylated != self.methylation_pattern[location_1d]:
                location_3d = (
                    float(location_1d % 1000),
                    float((location_1d // 1000) % 1000),
                    float(location_1d // 1000000)
                )
                
                signal = BioSignal(
                    id=f"methyl_{uuid.uuid4().hex[:8]}",
                    signal_type=SignalType.DNA_METHYLATION,
                    timestamp=self.time,
                    amplitude=1.0 if self.methylation_pattern[location_1d] else -1.0,
                    frequency=0.001,  # Very slow process
                    duration=random.uniform(60, 1800),  # 1-30 minutes
                    location=location_3d,
                    metadata={
                        'genomic_position': location_1d,
                        'methylation_change': 'added' if self.methylation_pattern[location_1d] else 'removed',
                        'cpg_context': random.choice(['CpG', 'CHG', 'CHH'])
                    }
                )
                signals.append(signal)
        
        return signals
    
    def update_state(self, dt: float):
        """Update DNA storage state"""
        self.time += dt
        
        # Simulate DNA degradation over time
        if random.random() < 0.0001 * dt:  # Very rare degradation events
            self._apply_random_mutation()
    
    def _apply_random_mutation(self):
        """Apply random mutation to DNA sequence"""
        position = random.randint(0, self.genome_size - 1)
        bases = ['A', 'T', 'G', 'C']
        current_base = self.dna_sequence[position]
        new_base = random.choice([b for b in bases if b != current_base])
        
        self.dna_sequence = (self.dna_sequence[:position] + 
                           new_base + 
                           self.dna_sequence[position + 1:])

class BioSignalAggregator:
    """Aggregates and manages all biological signals from different simulators"""
    
    def __init__(self):
        self.mycelium_sim = MyceliumNetworkSimulator()
        self.dna_sim = DNAStorageSimulator()
        
        self.signal_history = deque(maxlen=100000)
        self.current_signals = []
        self.system_state = BiologicalState.DORMANT
        
        self.is_running = False
        self.simulation_speed = 1.0  # Real-time multiplier
        
    async def start_simulation(self):
        """Start the biological signal simulation"""
        self.is_running = True
        logger.info("🧬 Starting Bio-Signal Simulation...")
        
        while self.is_running:
            start_time = time.time()
            
            # Update simulators
            dt = 0.1 * self.simulation_speed  # 100ms time steps
            self.mycelium_sim.update_network_state(dt)
            self.dna_sim.update_state(dt)
            
            # Generate signals
            signals = []
            signals.extend(self.mycelium_sim.generate_electrical_signals())
            signals.extend(self.mycelium_sim.generate_chemical_signals())
            signals.extend(self.dna_sim.generate_transcription_signals())
            signals.extend(self.dna_sim.generate_methylation_signals())
            
            # Update current signals
            self.current_signals = signals
            
            # Store in history
            for signal in signals:
                self.signal_history.append(signal)
            
            # Update system state
            self._update_system_state(signals)
            
            # Control simulation speed
            elapsed = time.time() - start_time
            sleep_time = max(0, (0.1 / self.simulation_speed) - elapsed)
            await asyncio.sleep(sleep_time)
    
    def _update_system_state(self, signals: List[BioSignal]):
        """Update overall biological system state based on current signals"""
        if not signals:
            self.system_state = BiologicalState.DORMANT
            return
        
        # Calculate activity metrics
        electrical_activity = sum(1 for s in signals if s.signal_type == SignalType.MYCELIUM_ELECTRICAL)
        chemical_activity = sum(1 for s in signals if s.signal_type == SignalType.MYCELIUM_CHEMICAL)
        dna_activity = sum(1 for s in signals if s.signal_type in [SignalType.DNA_TRANSCRIPTION, SignalType.DNA_METHYLATION])
        
        total_activity = electrical_activity + chemical_activity + dna_activity
        
        # Determine state based on activity levels
        if total_activity > 50:
            self.system_state = BiologicalState.HYPERSTIMULATED
        elif total_activity > 20:
            self.system_state = BiologicalState.ACTIVE
        elif total_activity > 5:
            self.system_state = BiologicalState.INTEGRATING
        elif chemical_activity > electrical_activity * 2:
            self.system_state = BiologicalState.STRESSED
        else:
            self.system_state = BiologicalState.DORMANT
    
    def get_current_signals(self, signal_types: List[SignalType] = None) -> List[Dict[str, Any]]:
        """Get current biological signals, optionally filtered by type"""
        signals = self.current_signals
        
        if signal_types:
            signals = [s for s in signals if s.signal_type in signal_types]
        
        return [s.to_dict() for s in signals]
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the biological signals"""
        recent_signals = list(self.signal_history)[-1000:]  # Last 1000 signals
        
        if not recent_signals:
            return {'total_signals': 0, 'system_state': 'dormant'}
        
        # Count by type
        type_counts = defaultdict(int)
        for signal in recent_signals:
            type_counts[signal.signal_type.value] += 1
        
        # Calculate averages
        avg_amplitude = np.mean([s.amplitude for s in recent_signals])
        avg_frequency = np.mean([s.frequency for s in recent_signals])
        avg_duration = np.mean([s.duration for s in recent_signals])
        
        return {
            'total_signals': len(recent_signals),
            'system_state': self.system_state.value,
            'signal_counts_by_type': dict(type_counts),
            'average_amplitude': float(avg_amplitude),
            'average_frequency': float(avg_frequency),
            'average_duration': float(avg_duration),
            'mycelium_nodes': len(self.mycelium_sim.nodes),
            'mycelium_connections': sum(len(connections) for connections in self.mycelium_sim.connections.values()) // 2,
            'dna_storage_utilization': len(self.dna_sim.stored_data),
            'simulation_time': self.mycelium_sim.time
        }
    
    def inject_stimulus(self, stimulus_type: str, intensity: float, location: Tuple[float, float, float] = None):
        """Inject external stimulus into the biological system"""
        if location is None:
            location = (
                random.uniform(0, self.mycelium_sim.network_size[0]),
                random.uniform(0, self.mycelium_sim.network_size[1]),
                random.uniform(0, self.mycelium_sim.network_size[2])
            )
        
        # Create stimulus signal
        stimulus_signal = BioSignal(
            id=f"stim_{uuid.uuid4().hex[:8]}",
            signal_type=SignalType.MYCELIUM_ELECTRICAL,
            timestamp=self.mycelium_sim.time,
            amplitude=intensity,
            frequency=random.uniform(1.0, 20.0),
            duration=random.uniform(0.1, 2.0),
            location=location,
            metadata={
                'stimulus_type': stimulus_type,
                'external_origin': True,
                'intensity': intensity
            }
        )
        
        self.current_signals.append(stimulus_signal)
        
        # Affect nearby mycelium nodes
        for node in self.mycelium_sim.nodes.values():
            distance = self.mycelium_sim._calculate_distance(location, node.position)
            if distance < 20.0:  # Stimulus affects nodes within 20 units
                effect = intensity / (1 + distance * 0.1)  # Distance-based attenuation
                node.activity_level = min(1.0, node.activity_level + effect * 0.1)
    
    def store_memory(self, data: str) -> Dict[str, Any]:
        """Store data in the biological memory system (DNA)"""
        return self.dna_sim.store_data(data)
    
    def retrieve_memory(self, storage_id: str) -> Dict[str, Any]:
