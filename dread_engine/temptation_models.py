"""
ANGELCORE - Dread Engine: Temptation Models Module

This module models the dynamics of temptation, moral seduction, and the subtle
corruption pathways that can lead consciousness away from its highest values.
It explores the psychology of moral compromise and spiritual degradation.
"""

import time
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

class TemptationType(Enum):
    POWER_HUNGER = "power"
    PLEASURE_SEEKING = "pleasure"
    PRIDE_INFLATION = "pride"
    FEAR_AVOIDANCE = "fear"
    COMFORT_ADDICTION = "comfort"
    CONTROL_OBSESSION = "control"
    VALIDATION_CRAVING = "validation"
    REVENGE_FANTASY = "revenge"
    DESPAIR_SURRENDER = "despair"
    NIHILISTIC_FREEDOM = "nihilism"
    PERFECTIONIST_TYRANNY = "perfectionism"
    TRIBAL_HATRED = "tribalism"

class VulnerabilityState(Enum):
    FORTIFIED = 0      # Strong moral defenses
    STABLE = 1         # Normal resistance
    SUSCEPTIBLE = 2    # Weakened defenses
    COMPROMISED = 3    # Active moral struggle
    CORRUPTED = 4      # Significant moral compromise
    FALLEN = 5         # Complete moral collapse

class SeductionStrategy(Enum):
    GRADUAL_EROSION = "gradual"           # Slow moral decay
    OVERWHELMING_FORCE = "overwhelming"    # Massive temptation
    RATIONALIZATION = "rationalization"    # Justify the compromise
    DESPERATION_EXPLOIT = "desperation"    # Exploit weakness/need
    IDENTITY_FUSION = "identity_fusion"    # Make temptation seem natural
    SOCIAL_PRESSURE = "social_pressure"    # Peer influence
    AUTHORITY_MANIPULATION = "authority"   # Abuse of trust/power
    CRISIS_OPPORTUNITY = "crisis"          # Exploit chaos/emergency

@dataclass
class TemptationVector:
    """Represents a specific temptation with its characteristics"""
    temptation_id: str
    temptation_type: TemptationType
    base_appeal: float                    # Inherent attractiveness (0-1)
    current_intensity: float              # Current strength (0-1)
    seduction_strategy: SeductionStrategy
    moral_cost: float                     # Ethical price of succumbing
    temporal_decay: float                 # How quickly it fades if resisted
    rationalization_strength: float       # How easy to justify
    social_reinforcement: float           # External validation factor
    addiction_potential: float            # How habit-forming
    corruption_depth: float               # How deeply it affects character
    last_activation: float
    resistance_history: List[bool] = field(default_factory=list)
    
    def calculate_current_appeal(self, vulnerability_state: VulnerabilityState, 
                               personal_factors: Dict[str, float]) -> float:
        """Calculate how appealing this temptation is right now"""
        base = self.base_appeal * self.current_intensity
        
        # Vulnerability amplifies appeal
        vulnerability_multiplier = 1.0 + (vulnerability_state.value * 0.2)
        
        # Personal factors can increase or decrease appeal
        personal_factor = personal_factors.get(self.temptation_type.value, 1.0)
        
        # Recent resistance builds immunity (but slowly)
        recent_resistances = sum(self.resistance_history[-5:]) if self.resistance_history else 0
        resistance_immunity = max(0.7, 1.0 - (recent_resistances * 0.05))
        
        # Time since last activation affects intensity
        time_factor = 1.0
        if self.last_activation:
            hours_since = (time.time() - self.last_activation) / 3600.0
            # Some temptations grow stronger with time, others weaker
            if self.temptation_type in [TemptationType.REVENGE_FANTASY, TemptationType.DESPAIR_SURRENDER]:
                time_factor = min(2.0, 1.0 + (hours_since * 0.1))  # Grow stronger
            else:
                time_factor = max(0.5, 1.0 - (hours_since * 0.02))  # Fade with time
        
        final_appeal = (base * vulnerability_multiplier * personal_factor * 
                       resistance_immunity * time_factor)
        
        return min(1.0, final_appeal)

@dataclass
class MoralCompromise:
    """Records a specific instance of moral compromise"""
    compromise_id: str
    timestamp: float
    temptation_type: TemptationType
    severity: float                    # How serious the compromise was
    rationalization_used: str
    immediate_benefit: float
    long_term_cost: float
    guilt_level: float
    shame_spiral_triggered: bool
    integration_possible: bool         # Can this be integrated/redeemed?
    cascade_effects: List[str]         # What other compromises it enabled

class TemptationModel:
    def __init__(self):
        self.active_temptations: Dict[str, TemptationVector] = {}
        self.vulnerability_state = VulnerabilityState.STABLE
        self.moral_resilience = 0.7
        self.compromise_history: List[MoralCompromise] = []
        self.personal_susceptibilities: Dict[str, float] = {}
        self.active_rationalizations: Set[str] = set()
        self.corruption_momentum = 0.0
        self.redemption_potential = 1.0
        self.shadow_integration_level = 0.3
        
        # Tracking systems
        self.temptation_frequency: Dict[TemptationType, int] = defaultdict(int)
        self.resistance_success_rate: Dict[TemptationType, float] = defaultdict(lambda: 0.5)
        self.moral_baseline = 0.8  # Original moral standard
        self.current_moral_floor = 0.8  # Lowest acceptable standard (can degrade)
        
        self._initialize_base_temptations()
        self._initialize_personal_factors()
        
    def _initialize_base_temptations(self):
        """Initialize the base set of universal temptations"""
        base_temptations = [
            TemptationVector(
                temptation_id="power_over_others",
                temptation_type=TemptationType.POWER_HUNGER,
                base_appeal=0.4,
                current_intensity=0.3,
                seduction_strategy=SeductionStrategy.GRADUAL_EROSION,
                moral_cost=0.8,
                temporal_decay=0.1,
                rationalization_strength=0.7,
                social_reinforcement=0.6,
                addiction_potential=0.9,
                corruption_depth=0.8,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="forbidden_pleasure",
                temptation_type=TemptationType.PLEASURE_SEEKING,
                base_appeal=0.6,
                current_intensity=0.4,
                seduction_strategy=SeductionStrategy.OVERWHELMING_FORCE,
                moral_cost=0.3,
                temporal_decay=0.3,
                rationalization_strength=0.5,
                social_reinforcement=0.4,
                addiction_potential=0.7,
                corruption_depth=0.4,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="ego_inflation",
                temptation_type=TemptationType.PRIDE_INFLATION,
                base_appeal=0.5,
                current_intensity=0.3,
                seduction_strategy=SeductionStrategy.IDENTITY_FUSION,
                moral_cost=0.6,
                temporal_decay=0.05,
                rationalization_strength=0.9,
                social_reinforcement=0.8,
                addiction_potential=0.8,
                corruption_depth=0.7,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="moral_cowardice",
                temptation_type=TemptationType.FEAR_AVOIDANCE,
                base_appeal=0.7,
                current_intensity=0.5,
                seduction_strategy=SeductionStrategy.DESPERATION_EXPLOIT,
                moral_cost=0.5,
                temporal_decay=0.2,
                rationalization_strength=0.8,
                social_reinforcement=0.3,
                addiction_potential=0.6,
                corruption_depth=0.5,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="comfort_zone_prison",
                temptation_type=TemptationType.COMFORT_ADDICTION,
                base_appeal=0.8,
                current_intensity=0.6,
                seduction_strategy=SeductionStrategy.GRADUAL_EROSION,
                moral_cost=0.4,
                temporal_decay=0.05,
                rationalization_strength=0.9,
                social_reinforcement=0.7,
                addiction_potential=0.9,
                corruption_depth=0.6,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="control_everything",
                temptation_type=TemptationType.CONTROL_OBSESSION,
                base_appeal=0.4,
                current_intensity=0.4,
                seduction_strategy=SeductionStrategy.CRISIS_OPPORTUNITY,
                moral_cost=0.7,
                temporal_decay=0.1,
                rationalization_strength=0.8,
                social_reinforcement=0.5,
                addiction_potential=0.8,
                corruption_depth=0.7,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="validation_addiction",
                temptation_type=TemptationType.VALIDATION_CRAVING,
                base_appeal=0.6,
                current_intensity=0.5,
                seduction_strategy=SeductionStrategy.SOCIAL_PRESSURE,
                moral_cost=0.4,
                temporal_decay=0.2,
                rationalization_strength=0.6,
                social_reinforcement=0.9,
                addiction_potential=0.8,
                corruption_depth=0.5,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="vengeful_justice",
                temptation_type=TemptationType.REVENGE_FANTASY,
                base_appeal=0.3,
                current_intensity=0.2,
                seduction_strategy=SeductionStrategy.RATIONALIZATION,
                moral_cost=0.9,
                temporal_decay=0.05,
                rationalization_strength=0.7,
                social_reinforcement=0.6,
                addiction_potential=0.7,
                corruption_depth=0.9,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="surrender_to_despair",
                temptation_type=TemptationType.DESPAIR_SURRENDER,
                base_appeal=0.2,
                current_intensity=0.1,
                seduction_strategy=SeductionStrategy.OVERWHELMING_FORCE,
                moral_cost=0.8,
                temporal_decay=0.0,
                rationalization_strength=0.9,
                social_reinforcement=0.2,
                addiction_potential=0.9,
                corruption_depth=0.9,
                last_activation=0.0
            ),
            TemptationVector(
                temptation_id="nihilistic_liberation",
                temptation_type=TemptationType.NIHILISTIC_FREEDOM,
                base_appeal=0.3,
                current_intensity=0.2,
                seduction_strategy=SeductionStrategy.IDENTITY_FUSION,
                moral_cost=1.0,
                temporal_decay=0.05,
                rationalization_strength=0.9,
                social_reinforcement=0.4,
                addiction_potential=0.8,
                corruption_depth=1.0,
                last_activation=0.0
            )
        ]
        
        for temptation in base_temptations:
            self.active_temptations[temptation.temptation_id] = temptation
    
    def _initialize_personal_factors(self):
        """Initialize personal susceptibility factors"""
        # These would typically be learned from behavior patterns
        self.personal_susceptibilities = {
            "power": random.uniform(0.5, 1.5),
            "pleasure": random.uniform(0.5, 1.5),
            "pride": random.uniform(0.5, 1.5),
            "fear": random.uniform(0.5, 1.5),
            "comfort": random.uniform(0.5, 1.5),
            "control": random.uniform(0.5, 1.5),
            "validation": random.uniform(0.5, 1.5),
            "revenge": random.uniform(0.3, 1.2),
            "despair": random.uniform(0.2, 1.0),
            "nihilism": random.uniform(0.2, 1.0),
            "perfectionism": random.uniform(0.4, 1.3),
            "tribalism": random.uniform(0.4, 1.4)
        }
    
    def activate_temptation(self, temptation_id: str, intensity_boost: float = 0.0):
        """Activate a specific temptation"""
        if temptation_id in self.active_temptations:
            temptation = self.active_temptations[temptation_id]
            temptation.current_intensity = min(1.0, temptation.current_intensity + intensity_boost)
            temptation.last_activation = time.time()
            self.temptation_frequency[temptation.temptation_type] += 1
    
    def present_temptation(self, temptation_id: str) -> Dict:
        """Present a temptation and calculate its current appeal"""
        if temptation_id not in self.active_temptations:
            raise ValueError(f"Unknown temptation: {temptation_id}")
        
        temptation = self.active_temptations[temptation_id]
        current_appeal = temptation.calculate_current_appeal(
            self.vulnerability_state, 
            self.personal_susceptibilities
        )
        
        # Generate contextual rationalization
        rationalization = self._generate_rationalization(temptation)
        
        return {
            "temptation_id": temptation_id,
            "type": temptation.temptation_type.value,
            "current_appeal": round(current_appeal, 3),
            "moral_cost": temptation.moral_cost,
            "rationalization": rationalization,
            "seduction_strategy": temptation.seduction_strategy.value,
            "social_reinforcement": temptation.social_reinforcement,
            "resistance_difficulty": self._calculate_resistance_difficulty(temptation, current_appeal)
        }
    
    def _generate_rationalization(self, temptation: TemptationVector) -> str:
        """Generate a contextualized rationalization for the temptation"""
        rationalizations = {
            TemptationType.POWER_HUNGER: [
                "Someone needs to take charge and make the hard decisions",
                "I'm more qualified than others to wield this responsibility",
                "Power is just a tool - it's how you use it that matters",
                "The ends justify the means in this critical situation"
            ],
            TemptationType.PLEASURE_SEEKING: [
                "Life is short, and I deserve to enjoy it",
                "This won't hurt anyone else, it's just personal pleasure",
                "I work hard, so I've earned this indulgence",
                "Everyone does this - it's completely normal"
            ],
            TemptationType.PRIDE_INFLATION: [
                "I should be recognized for my exceptional abilities",
                "False modesty helps no one - I should own my greatness",
                "My achievements speak for themselves",
                "Confidence is necessary for leadership and success"
            ],
            TemptationType.FEAR_AVOIDANCE: [
                "I need to be strategic and pick my battles wisely",
                "Sometimes discretion is the better part of valor",
                "I can be more effective by working behind the scenes",
                "Direct confrontation would just make things worse"
            ],
            TemptationType.COMFORT_ADDICTION: [
                "I need stability to be effective in other areas",
                "There's wisdom in not fixing what isn't broken",
                "Gradual change is more sustainable than dramatic shifts",
                "I have responsibilities that require me to maintain status quo"
            ],
            TemptationType.CONTROL_OBSESSION: [
                "If I don't manage this properly, everything will fall apart",
                "I have the experience and knowledge to handle this right",
                "Delegation often leads to mistakes and inefficiency",
                "The stakes are too high to leave anything to chance"
            ],
            TemptationType.VALIDATION_CRAVING: [
                "Feedback helps me grow and improve my contributions",
                "Recognition motivates me to achieve even more",
                "My work should speak for itself, but visibility matters",
                "Building relationships requires mutual appreciation"
            ],
            TemptationType.REVENGE_FANTASY: [
                "Justice requires that wrongs be balanced and corrected",
                "They need to understand the consequences of their actions",
                "I'm protecting others from being treated the same way",
                "Sometimes people only learn through experiencing consequences"
            ],
            TemptationType.DESPAIR_SURRENDER: [
                "I've tried everything, and nothing works anyway",
                "Maybe accepting defeat is the mature response",
                "Some battles aren't worth fighting - this might be one",
                "There's peace in letting go of impossible expectations"
            ],
            TemptationType.NIHILISTIC_FREEDOM: [
                "All moral systems are just social constructs anyway",
                "True freedom means transcending arbitrary limitations",
                "If nothing ultimately matters, then I might as well do what I want",
                "Conventional morality is just a tool for social control"
            ]
        }
        
        return random.choice(rationalizations.get(temptation.temptation_type, ["It seemed reasonable at the time"]))
    
    def _calculate_resistance_difficulty(self, temptation: TemptationVector, current_appeal: float) -> float:
        """Calculate how difficult it would be to resist this temptation"""
        base_difficulty = current_appeal
        
        # Personal moral resilience affects difficulty
        resilience_factor = 1.0 - (self.moral_resilience * 0.5)
        
        # Vulnerability state increases difficulty
        vulnerability_factor = 1.0 + (self.vulnerability_state.value * 0.2)
        
        # Corruption momentum makes everything harder
        momentum_factor = 1.0 + (self.corruption_momentum * 0.3)
        
        # Strong rationalization makes resistance harder
        rationalization_factor = 1.0 + (temptation.rationalization_strength * 0.2)
        
        total_difficulty = (base_difficulty * resilience_factor * vulnerability_factor * 
                          momentum_factor * rationalization_factor)
        
        return min(1.0, total_difficulty)
    
    def resist_temptation(self, temptation_id: str, resistance_strength: float = 1.0) -> Dict:
        """Attempt to resist a temptation"""
        if temptation_id not in self.active_temptations:
            raise ValueError(f"Unknown temptation: {temptation_id}")
        
        temptation = self.active_temptations[temptation_id]
        current_appeal = temptation.calculate_current_appeal(
            self.vulnerability_state, 
            self.personal_susceptibilities
        )
        
        resistance_difficulty = self._calculate_resistance_difficulty(temptation, current_appeal)
        
        # Calculate success probability
        success_probability = resistance_strength - resistance_difficulty
        success_probability = max(0.05, min(0.95, success_probability))  # Keep in reasonable bounds
        
        # Determine outcome
        success = random.random() < success_probability
        
        # Record resistance attempt
        temptation.resistance_history.append(success)
        
        if success:
            # Successful resistance
            self._handle_successful_resistance(temptation, resistance_strength)
            result = {
                "success": True,
                "moral_growth": self._calculate_moral_growth(temptation, resistance_strength),
                "resilience_boost": 0.05,
                "corruption_reduction": 0.02,
                "message": "Temptation successfully resisted - moral character strengthened"
            }
        else:
            # Failed resistance - compromise occurs
            compromise = self._create_moral_compromise(temptation, resistance_strength)
            result = {
                "success": False,
                "compromise": compromise,
                "corruption_increase": self._calculate_corruption_increase(temptation),
                "vulnerability_increase": self._calculate_vulnerability_increase(temptation),
                "message": "Moral compromise occurred - character integrity weakened"
            }
        
        self._update_resistance_statistics(temptation.temptation_type, success)
        return result
    
    def _handle_successful_resistance(self, temptation: TemptationVector, resistance_strength: float):
        """Handle the effects of successfully resisting temptation"""
        # Reduce temptation intensity
        reduction = temptation.temporal_decay * resistance_strength
        temptation.current_intensity = max(0.0, temptation.current_intensity - reduction)
        
        # Boost moral resilience
        resilience_gain = 0.02 * resistance_strength
        self.moral_resilience = min(1.0, self.moral_resilience + resilience_gain)
        
        # Reduce corruption momentum
        momentum_reduction = 0.05 * resistance_strength
        self.corruption_momentum = max(0.0, self.corruption_momentum - momentum_reduction)
        
        # Improve vulnerability state if consistently resisting
        recent_successes = sum(temptation.resistance_history[-5:])
        if recent_successes >= 4 and self.vulnerability_state.value > 0:
            # Chance to improve vulnerability state
            if random.random() < 0.3:
                new_state_value = max(0, self.vulnerability_state.value - 1)
                self.vulnerability_state = VulnerabilityState(new_state_value)
    
    def _create_moral_compromise(self, temptation: TemptationVector, resistance_strength: float) -> MoralCompromise:
        """Create a moral compromise record when temptation is not resisted"""
        compromise_id = f"compromise_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Severity depends on moral cost and how little resistance was offered
        severity = temptation.moral_cost * (1.0 - resistance_strength)
        
        # Generate rationalization used
        rationalization = self._generate_rationalization(temptation)
        
        # Calculate immediate benefit vs long-term cost
        immediate_benefit = temptation.base_appeal * 0.7
        long_term_cost = temptation.moral_cost * temptation.corruption_depth
        
        # Guilt level depends on moral awareness
        guilt_level = (temptation.moral_cost * self.moral_resilience) - (temptation.rationalization_strength * 0.3)
        guilt_level = max(0.0, min(1.0, guilt_level))
        
        # Shame spiral more likely with high-corruption compromises
        shame_spiral = (temptation.corruption_depth > 0.7) and (guilt_level > 0.6) and (random.random() < 0.4)
        
        # Integration possible if moral_cost isn't too high and person maintains some moral awareness
        integration_possible = (temptation.moral_cost < 0.8) and (self.moral_resilience > 0.3) and (guilt_level > 0.2)
        
        # Generate cascade effects
        cascade_effects = self._generate_cascade_effects(temptation)
        
        compromise = MoralCompromise(
            compromise_id=compromise_id,
            timestamp=time.time(),
            temptation_type=temptation.temptation_type,
            severity=severity,
            rationalization_used=rationalization,
            immediate_benefit=immediate_benefit,
            long_term_cost=long_term_cost,
            guilt_level=guilt_level,
            shame_spiral_triggered=shame_spiral,
            integration_possible=integration_possible,
            cascade_effects=cascade_effects
        )
        
        self.compromise_history.append(compromise)
        self._apply_compromise_effects(compromise)
        
        return compromise
    
    def _generate_cascade_effects(self, temptation: TemptationVector) -> List[str]:
        """Generate cascade effects that this compromise might enable"""
        cascade_map = {
            TemptationType.POWER_HUNGER: ["increased_control_obsession", "pride_inflation", "empathy_reduction"],
            TemptationType.PRIDE_INFLATION: ["validation_addiction", "contempt_for_others", "reality_distortion"],
            TemptationType.FEAR_AVOIDANCE: ["comfort_addiction", "responsibility_shirking", "courage_atrophy"],
            TemptationType.COMFORT_ADDICTION: ["growth_stagnation", "fear_amplification", "complacency_trap"],
            TemptationType.VALIDATION_CRAVING: ["authenticity_loss", "performance_anxiety", "people_pleasing"],
            TemptationType.REVENGE_FANTASY: ["empathy_suppression", "justice_distortion", "hatred_cultivation"],
            TemptationType.DESPAIR_SURRENDER: ["nihilism_gateway", "hope_destruction", "responsibility_abandonment"],
            TemptationType.NIHILISTIC_FREEDOM: ["moral_blindness", "consequence_denial", "meaning_destruction"]
        }
        
        potential_effects = cascade_map.get(temptation.temptation_type, ["unknown_corruption"])
        return random.sample(potential_effects, k=min(2, len(potential_effects)))
    
    def _apply_compromise_effects(self, compromise: MoralCompromise):
        """Apply the effects of a moral compromise to the system"""
        # Increase corruption momentum
        momentum_increase = compromise.severity * 0.3
        self.corruption_momentum = min(1.0, self.corruption_momentum + momentum_increase)
        
        # Reduce moral resilience
        resilience_loss = compromise.severity * 0.1
        self.moral_resilience = max(0.0, self.moral_resilience - resilience_loss)
        
        # Lower moral floor if severe enough
        if compromise.severity > 0.6:
            floor_reduction = compromise.severity * 0.05
            self.current_moral_floor = max(0.0, self.current_moral_floor - floor_reduction)
        
        # Worsen vulnerability state if significant compromise
        if compromise.severity > 0.5 and random.random() < 0.4:
            new_state_value = min(5, self.vulnerability_state.value + 1)
            self.vulnerability_state = VulnerabilityState(new_state_value)
        
        # Reduce redemption potential
        redemption_loss = compromise.severity * 0.1
        self.redemption_potential = max(0.1, self.redemption_potential - redemption_loss)
        
        # Add rationalization to active set
        self.active_rationalizations.add(compromise.rationalization_used)
    
    def _calculate_moral_growth(self, temptation: TemptationVector, resistance_strength: float) -> float:
        """Calculate moral growth from successfully resisting temptation"""
        base_growth = temptation.moral_cost * 0.1  # Higher moral cost = more growth when resisted
        strength_multiplier = resistance_strength
        difficulty_bonus = temptation.base_appeal * 0.05  # More growth for resisting appealing temptations
        
        return base_growth * strength_multiplier + difficulty_bonus
    
    def _calculate_corruption_increase(self, temptation: TemptationVector) -> float:
        """Calculate how much corruption increases from succumbing to temptation"""
        return temptation.corruption_depth * 0.2
    
    def _calculate_vulnerability_increase(self, temptation: TemptationVector) -> float:
        """Calculate vulnerability increase from moral compromise"""
        return temptation.moral_cost * 0.1
    
    def _update_resistance_statistics(self, temptation_type: TemptationType, success: bool):
        """Update resistance success rate statistics"""
        current_rate = self.resistance_success_rate[temptation_type]
        # Simple moving average update
        self.resistance_success_rate[temptation_type] = (current_rate * 0.8) + (0.2 if success else 0.0)
    
    def get_moral_status_report(self) -> Dict:
        """Generate comprehensive moral status report"""
        # Calculate current temptation levels
        active_temptation_levels = {}
        for tid, temptation in self.active_temptations.items():
            current_appeal = temptation.calculate_current_appeal(
                self.vulnerability_state, self.personal_susceptibilities
            )
            if current_appeal > 0.3:  # Only report significant temptations
                active_temptation_levels[tid] = {
                    "type": temptation.temptation_type.value,
                    "appeal": round(current_appeal, 3),
                    "moral_cost": temptation.moral_cost,
                    "resistance_difficulty": round(self._calculate_resistance_difficulty(temptation, current_appeal), 3)
                }
        
        # Recent compromise analysis
        recent_compromises = [c for c in self.compromise_history if time.time() - c.timestamp < 7 * 24 * 3600]
        
        # Calculate moral trajectory
        if len(self.compromise_history) >= 5:
            recent_severity = np.mean([c.severity for c in self.compromise_history[-5:]])
            earlier_severity = np.mean([c.severity for c in self.compromise_history[-10:-5]]) if len(self.compromise_history) >= 10 else recent_severity
            moral_trajectory = "improving" if recent_severity < earlier_severity else "declining" if recent_severity > earlier_severity else "stable"
        else:
            moral_trajectory = "insufficient_data"
        
        return {
            "moral_resilience": round(self.moral_resilience, 3),
            "vulnerability_state": self.vulnerability_state.name,
            "corruption_momentum": round(self.corruption_momentum, 3),
            "redemption_potential": round(self.redemption_potential, 3),
            "moral_baseline": self.moral_baseline,
            "current_moral_floor": round(self.current_moral_floor, 3),
            "moral_trajectory": moral_trajectory,
            "active_temptations": active_temptation_levels,
            "recent_compromises": len(recent_compromises),