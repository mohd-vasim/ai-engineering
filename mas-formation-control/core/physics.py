"""
Core physics and decentralized multi-agent formation control simulation.
Implements the Formation Control pattern from Chapter 5 (Figure 5.14).
"""
import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np


class Vector:
    """2D vector for drone positions, velocities, and offsets."""
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)

    def __repr__(self) -> str:
        return f"Vector({self.x:.2f}, {self.y:.2f})"

    def norm(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> 'Vector':
        n = self.norm()
        return Vector(self.x / n, self.y / n) if n > 1e-9 else Vector(0.0, 0.0)

    def clamp(self, max_speed: float) -> 'Vector':
        n = self.norm()
        if n > max_speed:
            return self.normalized() * max_speed
        return Vector(self.x, self.y)


def NORM(v: Vector) -> float:
    return v.norm()


class CircularObstacle:
    """Circular physical obstacle in the flight arena (e.g. tree, tower)."""
    def __init__(self, x: float, y: float, radius: float = 3.5, label: str = "Obstacle"):
        self.position = Vector(x, y)
        self.radius = float(radius)
        self.label = label


class DroneAgent:
    """
    Decentralized Drone Agent executing local control laws.
    Complies with Figure 5.14: Sense neighbor -> Calc error -> Adjust velocity.
    """
    def __init__(
        self,
        agent_id: str,
        initial_position: Vector,
        designated_offset: Vector,
        neighbor_id: Optional[str] = None,
        kp: float = 1.4,
        tolerance: float = 0.2,
        max_speed: float = 5.0,
        peer_safety_margin: float = 4.5,
    ):
        self.agent_id = agent_id
        self.position = initial_position
        self.velocity = Vector(0.0, 0.0)
        self.DESIGNATED_OFFSET = designated_offset
        self.neighbor_id = neighbor_id
        self.kp = kp
        self.TOLERANCE = tolerance
        self.max_speed = max_speed
        self.peer_safety_margin = peer_safety_margin
        self.is_avoiding = False
        self.trajectory: List[Tuple[float, float]] = [(self.position.x, self.position.y)]

    def update_control_loop(
        self,
        dt: float,
        swarm: Dict[str, 'DroneAgent'],
        obstacles: List[CircularObstacle]
    ) -> Dict[str, Any]:
        """
        Decentralized control loop step:
        1. Obstacle avoidance sub-loop (repulsion + tangential bypass)
        2. Neighbor sensing & desired position calculation
        3. Peer-yielding self-organization
        4. Proportional velocity adjustment
        """
        # --- Obstacle Avoidance Sub-Loop ---
        f_obs = Vector(0.0, 0.0)
        self.is_avoiding = False
        for obs in obstacles:
            delta = self.position - obs.position
            dist = delta.norm()
            safety_margin = obs.radius + 3.0
            if dist < safety_margin:
                self.is_avoiding = True
                repulsion = delta.normalized() * (safety_margin - dist) * 5.0
                tangent = Vector(-delta.normalized().y, delta.normalized().x) * 4.0
                f_obs = f_obs + repulsion + tangent

        # --- Formation Control Law ---
        pos_error_val = 0.0
        if self.neighbor_id and self.neighbor_id in swarm:
            neighbor_pos = swarm[self.neighbor_id].position
            desired_position = neighbor_pos + self.DESIGNATED_OFFSET
            position_error = desired_position - self.position
            pos_error_val = position_error.norm()

            # Peer-yielding force (self-organization)
            f_peer = Vector(0.0, 0.0)
            for peer_id, peer in swarm.items():
                if peer_id != self.agent_id:
                    p_delta = self.position - peer.position
                    p_dist = p_delta.norm()
                    if p_dist < self.peer_safety_margin and p_dist > 1e-3:
                        f_peer = f_peer + p_delta.normalized() * (self.peer_safety_margin - p_dist) * 3.0

            # Error threshold check (Figure 5.14)
            if NORM(position_error) > self.TOLERANCE:
                adj = position_error * self.kp
                if self.is_avoiding:
                    self.velocity = (self.velocity + (f_obs * 1.5 + adj * 0.2 + f_peer) * dt).clamp(self.max_speed)
                else:
                    self.velocity = (self.velocity + (adj + f_peer) * dt).clamp(self.max_speed)
            else:
                if self.is_avoiding:
                    self.velocity = (self.velocity + f_obs * dt).clamp(self.max_speed)
                else:
                    self.velocity = self.velocity * 0.98
        else:
            if self.is_avoiding:
                self.velocity = (self.velocity + f_obs * dt).clamp(self.max_speed)

        # Integration
        self.position = self.position + self.velocity * dt
        self.trajectory.append((round(self.position.x, 4), round(self.position.y, 4)))

        return {
            "agent_id": self.agent_id,
            "x": self.position.x,
            "y": self.position.y,
            "vx": self.velocity.x,
            "vy": self.velocity.y,
            "is_avoiding": self.is_avoiding,
            "position_error": pos_error_val,
        }


def build_swarm_formation(
    num_drones: int,
    formation_type: str = "grid",
    spacing_m: float = 10.0,
    leader_start: Tuple[float, float] = (20.0, 10.0),
) -> List[Tuple[str, Vector, Vector, Optional[str]]]:
    """
    Builds specifications for (agent_id, init_pos, designated_offset, neighbor_id).
    Supports 'line', 'v_shape', 'grid'.
    """
    leader_pos = Vector(leader_start[0], leader_start[1])
    agents = [("Drone_A", leader_pos, Vector(0.0, 0.0), None)]

    offsets: List[Vector] = []

    if formation_type == "line":
        for i in range(1, num_drones):
            offsets.append(Vector(0.0, -i * spacing_m))

    elif formation_type == "v_shape":
        for i in range(1, num_drones):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            offsets.append(Vector(side * row * spacing_m * 0.8, -row * spacing_m * 0.8))

    else:  # grid (default)
        cols = max(2, int(math.ceil(math.sqrt(num_drones - 1))))
        r, c = 0, 0
        for _ in range(1, num_drones):
            x_off = (c - (cols - 1) / 2.0) * spacing_m
            y_off = -(r + 1) * spacing_m
            offsets.append(Vector(x_off, y_off))
            c += 1
            if c >= cols:
                c = 0
                r += 1

    drone_names = [f"Drone_{chr(65 + i)}" for i in range(1, num_drones)]
    for name, offset in zip(drone_names, offsets):
        init_pos = leader_pos + offset
        agents.append((name, init_pos, offset, "Drone_A"))

    return agents


def simulate_swarm(
    num_drones: int = 5,
    formation_type: str = "grid",
    spacing_m: float = 10.0,
    sim_steps: int = 160,
    dt: float = 0.1,
    obstacles: Optional[List[CircularObstacle]] = None,
    leader_speed: float = 2.5,
    kp: float = 1.4,
    tolerance: float = 0.2,
) -> Dict[str, Any]:
    """
    Runs the complete physics simulation and returns step logs and aggregated metrics.
    """
    if obstacles is None:
        obstacles = [CircularObstacle(x=20.0, y=40.0, radius=3.5, label="Tree Obstacle")]

    agent_specs = build_swarm_formation(num_drones, formation_type, spacing_m)
    swarm: Dict[str, DroneAgent] = {}
    for agent_id, init_pos, offset, neighbor_id in agent_specs:
        swarm[agent_id] = DroneAgent(
            agent_id, init_pos, offset, neighbor_id, kp=kp, tolerance=tolerance
        )

    leader = swarm["Drone_A"]
    step_logs = []

    for step in range(sim_steps):
        # Leader advances at survey speed
        leader.velocity = Vector(0.0, leader_speed)
        leader.position = leader.position + leader.velocity * dt
        leader.trajectory.append((round(leader.position.x, 4), round(leader.position.y, 4)))

        # Followers update via decentralized control loop
        agent_statuses = {
            "Drone_A": {
                "agent_id": "Drone_A",
                "x": leader.position.x,
                "y": leader.position.y,
                "vx": leader.velocity.x,
                "vy": leader.velocity.y,
                "is_avoiding": False,
                "position_error": 0.0,
            }
        }
        for agent_id, agent in swarm.items():
            if agent_id != "Drone_A":
                status = agent.update_control_loop(dt, swarm, obstacles)
                agent_statuses[agent_id] = status

        # Formation errors across followers
        errors = []
        for agent_id, agent in swarm.items():
            if agent.neighbor_id and agent.neighbor_id in swarm:
                tgt = swarm[agent.neighbor_id].position + agent.DESIGNATED_OFFSET
                errors.append((tgt - agent.position).norm())

        # Inter-agent clearances
        agents_list = list(swarm.values())
        min_dist = float('inf')
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                d = (agents_list[i].position - agents_list[j].position).norm()
                if d < min_dist:
                    min_dist = d

        mean_err = float(np.mean(errors)) if errors else 0.0
        drone_c_avoiding = float(swarm.get("Drone_C", swarm["Drone_A"]).is_avoiding)

        step_logs.append({
            "step": step,
            "mean_error": round(mean_err, 6),
            "min_clearance": round(min_dist, 6),
            "drone_c_avoiding": drone_c_avoiding,
            "positions": {aid: [round(a.position.x, 4), round(a.position.y, 4)] for aid, a in swarm.items()},
            "statuses": agent_statuses,
        })

    # Trajectories record
    trajectories = {aid: list(a.trajectory) for aid, a in swarm.items()}
    errors_series = [log["mean_error"] for log in step_logs]
    clearance_series = [log["min_clearance"] for log in step_logs]

    metrics = {
        "num_drones": num_drones,
        "formation_type": formation_type,
        "spacing_m": spacing_m,
        "max_deviation_meters": max(errors_series),
        "final_formation_error_meters": errors_series[-1],
        "minimum_clearance_meters": min(clearance_series),
        "sim_steps": sim_steps,
        "dt": dt,
    }

    return {
        "metrics": metrics,
        "step_logs": step_logs,
        "trajectories": trajectories,
        "obstacles": [{"x": obs.position.x, "y": obs.position.y, "radius": obs.radius, "label": obs.label} for obs in obstacles],
    }
