# Functional Design Specification (FDS): Contract-Net Marketplace Pattern

## 1. System Overview & Problem Statement

* **Context**: The system operates in a distributed environment featuring a diverse, heterogeneous, and dynamic pool of specialist agents whose capabilities overlap, but whose availability, cost, and performance characteristics vary dynamically. Static routing logic is brittle and inefficient in this environment because it cannot account for real-time load or specific task nuances.
* **Problem**: The core challenge is assigning a task to the most suitable agent when the optimal choice depends on dynamic factors (availability, cost, confidence) that are only known at runtime. This requires balancing specialization against routing overhead, competitive bidding against coordination cost, and exploration goals against service-level agreements (SLAs).
* **Solution**: Implement the **Contract-Net Marketplace** pattern (a market-based negotiation mechanism/Contract-Net Protocol) to move beyond hardcoded logic to a dynamic, market-driven model for task assignment.

---

## 2. Roles and Main Components

The system architecture consists of two primary roles:

### 2.1 The Solicitor

* **Responsibilities**:
* Manages the auction and negotiation process.
* Broadcasts task announcements to all available subscribers/potential workers.
* Collects incoming bids.
* Acts as an awarder, evaluating incoming bids based on a utility function.
* Awards the contract, assigning the task to the agent yielding the highest utility score.



### 2.2 The BidderAgent

* **Responsibilities**:
* Represents a specialized resource/worker.
* Receives task announcements from the Solicitor.
* Evaluates the task announcement to determine capability (`can_handle(task)`). If incapable, returns a refusal (`None`).
* Assesses capability to calculate a confidence score for the task (`assess_capability(task)`).
* Estimates compute/operational cost (`estimate_compute_cost(task)`).
* Responds to the announcement by submitting a formal bid containing its capability/confidence score, cost, and estimated time of arrival (ETA).
* Executes the awarded contract (`execute_contract(task)`) if chosen by the Solicitor.



---

## 3. Core Functional Workflows

### 3.1 Task Allocation Sequence

1. **Announcement**: The Solicitor receives a task and broadcasts a task announcement (e.g., "Task: Train Model X. Constraint: Max cost $100.") to potential workers.
2. **Bidding**: Bidders evaluate the announcement and return a formal bid containing their metrics (e.g., AWS_Agent bids $90, ETA 2 hours; Azure_Agent bids $85, ETA 2.5 hours; OnPrem_Agent bids $10, ETA 12 hours).
3. **Award**: The Solicitor weighs the variables (e.g., time versus money via the utility function) and awards the contract to the agent providing the best balance. The selected agent then executes the contract.

---

## 4. System Consequences & Execution Guardrails

### 4.1 Consequences

* **Pros**:
* **Adaptive Selection**: The system dynamically adapts to changing agent availability and capabilities without code changes.
* **High Utilization**: Decouples the requester from the provider, ensuring work flows to the agents best suited for it at that exact moment.


* **Cons**:
* **Auction Latency**: The negotiation process introduces computational and time overhead before work even begins.
* **Risk of Gaming**: Without incentives for truthful bidding, agents may overstate their confidence to win tasks.



### 4.2 Implementation Guidance & Guardrails

* **Target Use Case**: Use this pattern when operating with a large, variable toolset or when optimizing for dynamic factors like cost or speed.
* **Anti-Patterns**: Avoid using this pattern for fixed, predictable workflows where static routing (such as intent-aware routing) is simpler and faster.
* **Strict Deadlines**: To prevent infinite waiting, the Solicitor must enforce strict deadlines for receiving bids.
* **Reputation Score**: Consider implementing a "reputation score" for bidders to penalize agents that win contracts but fail to deliver quality results.
* **Enterprise Trade-offs**: Market-based allocation optimizes efficiency and cost in dynamic environments with multiple specialists. However, in enterprise systems where agents perform high-stakes or risky actions, efficiency must be balanced with absolute stability and fault tolerance.