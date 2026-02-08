# UCR Semantic Reference

## Purpose

This document defines the precise semantic meaning of every Force token, Object token, and their valid combinations (anchors) in the Slipstream v3 Universal Concept Reference. It is intended as a ground-truth reference for:

- **DPO (Direct Preference Optimization)**: Determining which quantization is "chosen" (correct) vs "rejected" (incorrect) for a given natural language input.
- **RLHF training**: Providing reward signal for whether a model selected the right Force-Object pair.
- **Annotation guidelines**: Giving human labelers unambiguous criteria for judging quantization quality.

All definitions are normative for Slipstream v3. The 12 Force tokens form a closed vocabulary. The Object tokens listed here are the 31 core objects; installations may add extensions in the 0x8000-0xFFFF address range.

---

## 1. Force Tokens (12)

Force tokens classify the **speech act type** -- what the sender is doing by sending the message. They answer: "What kind of communicative action is this?"

The Force vocabulary is closed. No new Force tokens may be added at runtime.

### 1.1 Observe

**Definition**: The sender reports something it has passively detected or monitored. The sender is a witness, not a participant -- it is surfacing raw observations about the world.

**Speech act mapping**: Assertive (constative). The sender asserts a fact about something it perceived. Closest to Searle's "representatives" -- the speaker represents a state of affairs.

**When to use**:
- A monitoring agent detects a state change in a system it watches
- A sensor reports current readings
- An agent notices an error condition in another component
- A watcher reports that a deployment has changed

**When NOT to use**:
- The sender computed or produced the result (use Inform)
- The sender is asking about state rather than reporting it (use Ask)
- The sender is reporting its own status or progress (use Inform -- Observe is for external observations)

**Example natural language inputs**:
- "I noticed the database connection pool dropped to 2 active connections"
- "The build server state is currently idle"
- "Detected a configuration change in the auth module"
- "Environment variable DB_HOST changed from prod1 to prod2"
- "I see that the error rate spiked to 5% in the last minute"

---

### 1.2 Inform

**Definition**: The sender actively shares information it possesses -- results, status, progress, or completion notices. Unlike Observe, the sender is a participant delivering information it generated or owns.

**Speech act mapping**: Assertive (informative). The sender informs the receiver of facts the sender is responsible for or has produced. This is the most common Force for status reporting, result delivery, and progress updates.

**When to use**:
- Reporting the result of a computation or task
- Giving a status update on work the sender is doing
- Announcing task completion
- Reporting that the sender is blocked on something
- Sharing progress on an ongoing task

**When NOT to use**:
- Reporting something the sender passively observed in an external system (use Observe)
- Asking for information (use Ask)
- Making a judgment or evaluation about work quality (use Eval)

**Example natural language inputs**:
- "The test suite passed: 142 tests, 0 failures"
- "I am currently working on the auth refactor"
- "Task complete: the API endpoint is deployed"
- "I am blocked waiting for database credentials"
- "Progress update: 3 of 5 migration steps finished"
- "FYI, the deploy finished successfully"
- "Heads up: the staging server is using 90% memory"

---

### 1.3 Ask

**Definition**: The sender seeks information, clarification, or permission from the receiver. The sender does not know something and needs the receiver to supply it. This is a question, not a command.

**Speech act mapping**: Directive (interrogative). The sender directs the receiver to provide information. Maps to Searle's "directives" subcategory of questions -- the desired response is informational, not an action.

**When to use**:
- Requesting clarification about ambiguous instructions
- Querying the current status of something
- Asking whether the sender has permission to proceed
- Asking about resource availability

**When NOT to use**:
- Telling someone to do something (use Request)
- Reporting status to someone who asked (use Inform)
- Suggesting a course of action (use Propose)
- The sender already knows the answer and is sharing it (use Inform)

**Example natural language inputs**:
- "What do you mean by 'optimize the query'?"
- "How is the migration progressing?"
- "Can I deploy to production now?"
- "Is the GPU cluster available for training?"
- "What is the current status of the auth module?"

---

### 1.4 Request

**Definition**: The sender directs the receiver to perform a specific action. This is a command or solicitation for work, not a question.

**Speech act mapping**: Directive (imperative). The sender wants the receiver to do something. Maps to Searle's "directives" -- the illocutionary point is to get the hearer to do something.

**When to use**:
- Assigning a task to another agent
- Asking for a code review (the action is "review", not a question)
- Asking for help with a problem (the action is "assist")
- Requesting cancellation of an ongoing operation
- Requesting resource allocation
- Requesting a plan be created

**When NOT to use**:
- Asking a question to get information (use Ask)
- Suggesting an approach without directing action (use Propose)
- Committing to do the work yourself (use Commit)
- The sender is doing the task itself (use Commit or Inform)

**Example natural language inputs**:
- "Please implement the caching layer for the user service"
- "Review the pull request for auth changes"
- "I need help debugging the memory leak"
- "Cancel the deployment to staging"
- "Allocate 4 GPUs for the training run"
- "Create a plan for the database migration"

---

### 1.5 Propose

**Definition**: The sender suggests a course of action without directing the receiver to take it. The sender is offering an idea for consideration, not issuing a command.

**Speech act mapping**: Commissive-directive hybrid. The sender proposes a joint or unilateral action for consideration. This is weaker than Request (no obligation on the receiver) and weaker than Commit (no commitment from the sender).

**When to use**:
- Suggesting an alternative approach to a problem
- Offering a plan for consideration and discussion
- Recommending a change that needs approval
- Suggesting a rollback when something went wrong

**When NOT to use**:
- Telling someone to do something (use Request)
- Committing to do it yourself (use Commit)
- Evaluating work that was done (use Eval)
- The decision has already been made (use Request or Commit)

**Example natural language inputs**:
- "I suggest we use Redis instead of Memcached for the session store"
- "How about we split the monolith into three services?"
- "I propose we roll back the last deployment"
- "We could use a different authentication provider"
- "My recommendation is to add an index on the users table"

---

### 1.6 Commit

**Definition**: The sender pledges to perform an action or allocate a resource. The sender is making a binding commitment, not requesting or proposing.

**Speech act mapping**: Commissive. The sender commits themselves to a future course of action. Maps directly to Searle's "commissives" -- promises, pledges, and guarantees.

**When to use**:
- Taking ownership of a task
- Promising to deliver by a deadline
- Allocating resources (compute, budget, personnel)
- Confirming that the sender will do the work

**When NOT to use**:
- Requesting someone else do the work (use Request)
- Suggesting something without committing (use Propose)
- Reporting that work is already done (use Inform with Complete)
- Accepting someone else's proposal (use Accept)

**Example natural language inputs**:
- "I will implement the caching layer by Friday"
- "I am taking ownership of the auth module refactor"
- "Allocating 8 GPUs to the training cluster"
- "I commit to delivering the API docs by end of sprint"
- "On it -- I will handle the database migration"

---

### 1.7 Eval

**Definition**: The sender makes a qualitative judgment about work, a proposal, or a deliverable. The sender is assessing quality or completeness, not just reporting facts.

**Speech act mapping**: Assertive (evaluative). The sender expresses a judgment. This is a specialized assertive that carries normative weight -- it is not neutral reporting but a quality determination.

**When to use**:
- Approving submitted work after review
- Indicating that work needs revision
- Confirming that reviewed work is complete and meets standards
- Rendering a verdict on a pull request or deliverable

**When NOT to use**:
- Accepting or rejecting a proposal or request (use Accept/Reject)
- Simply reporting that a task finished (use Inform Complete)
- Asking questions during a review (use Ask)
- Requesting changes without having evaluated (use Request)

**Example natural language inputs**:
- "LGTM, approved for merge"
- "This needs revision: the error handling is missing edge cases"
- "Looks good to me, ship it"
- "The implementation is complete and meets all acceptance criteria"
- "Changes needed: the API response format does not match the spec"

---

### 1.8 Meta

**Definition**: The sender performs a protocol-level or coordination action that is about the communication process itself, not about the work content. Acknowledgments, handoffs, synchronization pings, escalations, and aborts are all Meta.

**Speech act mapping**: Expressive / Declarative (meta-communicative). These acts manage the conversation and coordination protocol rather than advancing the work directly.

**When to use**:
- Acknowledging receipt of a message
- Pinging another agent to confirm it is alive
- Handing off responsibility to another agent
- Escalating an issue to a higher authority
- Aborting an operation due to emergency
- Deferring a decision to a later time

**When NOT to use**:
- Accepting a proposal (use Accept -- that is a substantive response, not protocol management)
- Reporting an error condition (use Error or Observe Error)
- Providing a status update (use Inform)
- Just saying "yes" to a request (use Accept)

**Example natural language inputs**:
- "Acknowledged, I received the task assignment"
- "Ping -- are you still online?"
- "Handing this off to the security team"
- "This is above my authority, escalating to the engineering manager"
- "Emergency: abort the production deployment immediately"
- "Let's revisit this decision next sprint"

---

### 1.9 Accept

**Definition**: The sender agrees to a proposal, request, or plan. This is an affirmative response to something another agent put forward.

**Speech act mapping**: Commissive (responsive). The sender binds themselves to the terms of what was proposed or requested. Accept is always a response to a prior message.

**When to use**:
- Agreeing to a proposal
- Confirming a request will be fulfilled
- Accepting with conditions ("yes, but only if...")
- Giving the go-ahead

**When NOT to use**:
- Approving work quality after review (use Eval Approve)
- Committing to a task you initiated yourself (use Commit)
- Acknowledging receipt without agreeing to anything (use Meta Ack)
- Simply saying "understood" (use Meta Ack)

**Example natural language inputs**:
- "Yes, I agree with that plan"
- "Accepted -- we will proceed with the Redis approach"
- "Confirmed, go ahead"
- "Yes, but only if we add monitoring first"
- "Agreed, provided that we keep the fallback path"

---

### 1.10 Reject

**Definition**: The sender declines or refuses a proposal, request, or plan. This is a negative response to something another agent put forward.

**Speech act mapping**: Commissive (negative responsive). The sender refuses to be bound by the proposed terms. Reject is always a response to a prior message.

**When to use**:
- Declining a proposal
- Refusing a request
- Saying no to a plan
- Disagreeing with a suggested approach

**When NOT to use**:
- Evaluating work as needing revision (use Eval NeedsWork -- that is a quality judgment, not a refusal)
- Reporting an error (use Error)
- Aborting an operation (use Meta Abort)
- Giving negative feedback on work quality (use Eval)

**Example natural language inputs**:
- "No, I do not agree with that approach"
- "Rejected -- the timeline is not feasible"
- "I decline that request"
- "Disagree, we should not move forward with this"
- "I refuse to proceed without proper testing"

---

### 1.11 Error

**Definition**: The sender reports a system-level error condition. This is a technical failure report, not a judgment or observation.

**Speech act mapping**: Assertive (error-reporting). The sender asserts that a technical failure has occurred. This is distinct from Observe (which is general-purpose monitoring) because Error specifically signals a fault condition that requires attention.

**When to use**:
- A computation failed with an exception
- An operation timed out
- A required resource is unavailable
- A permission check failed
- Input validation failed

**When NOT to use**:
- Observing an error in another system that the sender monitors (use Observe Error)
- Disagreeing with a proposal (use Reject)
- Evaluating work as deficient (use Eval NeedsWork)
- Reporting a blocker that is not a technical error (use Inform Blocked)

**Example natural language inputs**:
- "Error: database connection failed with timeout after 30s"
- "The API call to the payment provider timed out"
- "Resource unavailable: no GPU instances in the pool"
- "Permission denied: service account lacks write access"
- "Validation failed: the input schema does not match"

---

### 1.12 Fallback

**Definition**: The sender's intent cannot be quantized into any Force-Object pair with sufficient confidence. The raw content is stored out-of-band and referenced by a pointer. Fallback is a safety valve, not a normal classification.

**Speech act mapping**: None. Fallback explicitly signals the absence of a clean speech-act classification. The content must be retrieved via the fallback reference for interpretation.

**When to use**:
- The natural language input does not match any Force-Object combination
- The quantizer confidence is below threshold
- The content is inherently unstructured (e.g., a long free-form discussion)

**When NOT to use**:
- Fallback should never be chosen when a valid Force-Object pair exists. It is a last resort, not a default. If any anchor matches with reasonable confidence, use that anchor.

**Example natural language inputs**:
- Complex multi-topic messages that span several intents
- Highly ambiguous statements with no clear speech act
- Raw data dumps or log outputs that contain no clear intent

---

## 2. Object Tokens (31 Core)

Object tokens classify the **domain concept** -- what the message is about. They answer: "What thing is being acted upon?"

Object tokens are extensible. Core objects (listed here) are immutable per version. Extensions can be added at runtime in the 0x8000-0xFFFF address range.

### 2.1 State

**Definition**: The current condition or configuration of a system, component, or environment at a point in time.

**Common Force pairings**: Observe (report observed state), Ask (query what the state is)

**Example inputs**: "The system state is healthy", "Current memory usage is 74%", "The environment is configured for production"

---

### 2.2 Change

**Definition**: A detected modification or transition from one state to another.

**Common Force pairings**: Observe (detected a change), Propose (suggest making a change)

**Example inputs**: "The config was modified", "Detected a version bump in the dependency", "I suggest we change the retry logic"

---

### 2.3 Error

**Definition**: A fault condition, exception, or failure in a system or process.

**Common Force pairings**: Observe (witnessed an error), Error (reporting own error)

**Example inputs**: "Null pointer exception in the parser module", "The build failed on the CI server", "Exception thrown during data migration"

---

### 2.4 Result

**Definition**: The output or outcome of a computation, task, or analysis.

**Common Force pairings**: Inform (sharing a result that was produced)

**Example inputs**: "The benchmark shows 340ms p99 latency", "Test results: 98% pass rate", "Query returned 1,247 matching records"

---

### 2.5 Status

**Definition**: The current operational state of a task, process, or agent. Status is about ongoing work, while State is about system configuration.

**Common Force pairings**: Inform (giving a status update), Ask (querying status)

**Example inputs**: "Deployment is in progress", "What is the status of the migration?", "Currently running integration tests"

---

### 2.6 Complete

**Definition**: The successful finishing of a task or process. Signals that work is done.

**Common Force pairings**: Inform (announcing completion), Eval (confirming work meets completion criteria)

**Example inputs**: "The refactor is finished", "All migration steps completed successfully", "Done with the code review"

---

### 2.7 Blocked

**Definition**: A dependency or impediment preventing work from continuing. The sender cannot proceed.

**Common Force pairings**: Inform (reporting being blocked), Eval (determining something is blocked)

**Example inputs**: "Waiting on the API credentials from the security team", "Blocked by the database lock", "Cannot proceed until the schema migration runs"

---

### 2.8 Progress

**Definition**: An incremental update on work that is underway but not yet complete.

**Common Force pairings**: Inform (sharing progress)

**Example inputs**: "3 of 5 endpoints migrated", "Working on the authentication module", "Making progress on the test coverage"

---

### 2.9 Clarify

**Definition**: A request for disambiguation or further explanation of something that is unclear.

**Common Force pairings**: Ask (requesting clarification)

**Example inputs**: "What do you mean by 'optimize'?", "Which database are you referring to?", "Can you clarify the acceptance criteria?"

---

### 2.10 Permission

**Definition**: Authorization to proceed with an action that requires approval.

**Common Force pairings**: Ask (requesting permission), Error (permission denied)

**Example inputs**: "May I deploy to production?", "Can I merge this to main?", "Permission denied on the staging server"

---

### 2.11 Resource

**Definition**: A computational, infrastructure, or organizational resource (CPU, memory, budget, personnel, etc.).

**Common Force pairings**: Ask (querying availability), Request (requesting allocation), Commit (pledging resources), Error (resource unavailable)

**Example inputs**: "Allocate 4 GPUs", "Is the test cluster available?", "Resource pool exhausted", "Committing 2 engineers to the project"

---

### 2.12 Task

**Definition**: A discrete unit of work to be executed.

**Common Force pairings**: Request (assigning a task), Commit (taking on a task)

**Example inputs**: "Implement the caching layer", "Run the integration tests", "Execute the database migration"

---

### 2.13 Plan

**Definition**: A strategy, design, or sequence of steps for achieving a goal.

**Common Force pairings**: Request (asking for a plan), Propose (suggesting a plan)

**Example inputs**: "Create a plan for the API migration", "Here is my proposal for the release schedule", "How should we approach the refactor?"

---

### 2.14 Review

**Definition**: Examination and evaluation of work, code, or a deliverable.

**Common Force pairings**: Request (asking for a review), Eval (giving review results)

**Example inputs**: "Review the pull request", "Check the auth implementation", "Look at the API design doc"

---

### 2.15 Help

**Definition**: Assistance, guidance, or support with a problem or task.

**Common Force pairings**: Request (asking for help)

**Example inputs**: "I need help with the memory leak", "Can you assist with the deployment?", "I need guidance on the error handling approach"

---

### 2.16 Cancel

**Definition**: Termination of an ongoing or planned operation.

**Common Force pairings**: Request (requesting cancellation)

**Example inputs**: "Cancel the deployment", "Stop the training run", "Nevermind, do not proceed with that task"

---

### 2.17 Priority

**Definition**: The urgency or ordering of a task relative to others.

**Common Force pairings**: Request (asking for priority change)

**Example inputs**: "This needs to be expedited", "Raise the priority of the security fix", "Mark this as urgent"

---

### 2.18 Alternative

**Definition**: A different approach or option compared to what was previously discussed.

**Common Force pairings**: Propose (suggesting an alternative)

**Example inputs**: "Instead of REST, we could use GraphQL", "Another approach would be to use event sourcing", "Here is an alternative to the proposed architecture"

---

### 2.19 Rollback

**Definition**: Reverting a system, deployment, or change to a prior state.

**Common Force pairings**: Propose (suggesting a rollback)

**Example inputs**: "I suggest we revert the last deployment", "We should undo the schema change", "Roll back to version 2.3.1"

---

### 2.20 Deadline

**Definition**: A time-bound commitment or due date for a deliverable.

**Common Force pairings**: Commit (pledging to a deadline)

**Example inputs**: "I will deliver by Friday", "The ETA is end of sprint", "We need to ship by March 1"

---

### 2.21 Approve

**Definition**: A positive quality judgment -- the work meets standards.

**Common Force pairings**: Eval (approving after evaluation)

**Example inputs**: "LGTM", "Approved for merge", "Looks good, ship it"

---

### 2.22 NeedsWork

**Definition**: A quality judgment indicating that the work requires revision before it can be accepted.

**Common Force pairings**: Eval (indicating revisions needed)

**Example inputs**: "Needs revision: missing error handling", "Changes needed on the API response format", "Almost there but the tests are incomplete"

---

### 2.23 Ack

**Definition**: Acknowledgment of receipt. Confirms a message was received, without agreeing or disagreeing.

**Common Force pairings**: Meta (protocol-level acknowledgment)

**Example inputs**: "Got it", "Acknowledged", "Received, understood"

---

### 2.24 Sync

**Definition**: A synchronization or liveness check between agents.

**Common Force pairings**: Meta (protocol-level coordination)

**Example inputs**: "Ping", "Are you still there?", "Heartbeat check"

---

### 2.25 Handoff

**Definition**: Transfer of responsibility or ownership from one agent to another.

**Common Force pairings**: Meta (coordination handoff)

**Example inputs**: "Passing this to the security team", "Your turn to handle the deployment", "Transferring ownership of the auth module"

---

### 2.26 Escalate

**Definition**: Elevation of an issue to a higher authority because it exceeds the sender's scope or ability.

**Common Force pairings**: Meta (protocol-level escalation)

**Example inputs**: "This is above my authority, escalating to the manager", "Need a senior engineer to decide on this", "Raising this to the architecture board"

---

### 2.27 Abort

**Definition**: Emergency termination of an operation. Stronger than Cancel -- implies urgency and potential danger.

**Common Force pairings**: Meta (protocol-level emergency stop)

**Example inputs**: "Emergency stop: halt the production deployment", "Abort -- critical failure detected", "Halt all operations immediately"

---

### 2.28 Condition

**Definition**: A qualifier or stipulation attached to an acceptance. Used for conditional agreement.

**Common Force pairings**: Accept (conditional acceptance)

**Example inputs**: "Yes, but only if we add monitoring", "Agreed, provided we keep the rollback path", "Accepted on the condition that tests pass"

---

### 2.29 Defer

**Definition**: Postponement of a decision or action to a later time.

**Common Force pairings**: Meta (deferring a decision)

**Example inputs**: "Let's revisit this next sprint", "Postponing the decision until we have more data", "Not now, we can address this later"

---

### 2.30 Timeout

**Definition**: A failure caused by an operation exceeding its time limit.

**Common Force pairings**: Error (timeout failure)

**Example inputs**: "The API call timed out after 30 seconds", "Database query exceeded the 5-second limit", "Connection attempt timed out"

---

### 2.31 Validation

**Definition**: A failure caused by input or data not conforming to expected format or rules.

**Common Force pairings**: Error (validation failure)

**Example inputs**: "Invalid JSON payload", "The request body is missing required fields", "Schema validation failed on the config file"

---

### 2.32 Generic

**Definition**: A catch-all object used when no specific domain concept applies. Used with Accept, Reject, Error, and Fallback when the Force alone carries the semantic weight.

**Common Force pairings**: Accept (general acceptance), Reject (general rejection), Error (general error), Fallback (unquantizable content)

**Example inputs**: Varies. Generic is the default when the Object cannot be more specifically classified.

---

## 3. Core Anchor Table (45 Anchors)

The following table lists every core UCR anchor. Coords are (ACTION, POLARITY, DOMAIN, URGENCY) on 0-7 scales.

| Index | Force | Object | Canonical | Coords | Semantic Meaning |
|-------|-------|--------|-----------|--------|-----------------|
| 0x0001 | Observe | State | Report current state | (0,4,2,3) | The sender has passively monitored a system and is reporting what it sees right now. This is a snapshot observation, not a change notification. Used when an agent watches infrastructure or environment and surfaces current conditions. |
| 0x0002 | Observe | Change | Report detected change | (0,4,2,4) | The sender detected a transition or modification in something it monitors. Unlike Observe State, this signals that something is different from before. Used when a watcher notices a config change, deployment, or state transition. |
| 0x0003 | Observe | Error | Report observed error | (0,2,6,6) | The sender witnessed an error condition in a system it monitors. The sender did not cause the error and is not the failing component -- it is a third-party observer reporting a fault. Higher urgency than other Observe anchors due to the error nature. |
| 0x0010 | Inform | Result | Share computed result | (1,5,2,3) | The sender produced or computed something and is delivering the output. This is active information sharing of work product -- test results, query outputs, benchmark numbers. The sender owns or generated the result. |
| 0x0011 | Inform | Status | Provide status update | (1,4,0,3) | The sender reports on the current state of its own work or responsibilities. Unlike Observe State (external), this is a self-report. Used for routine check-ins on task progress or operational status. |
| 0x0012 | Inform | Complete | Report task completion | (1,6,0,4) | The sender announces that a task it was working on is finished. Positive polarity because completion is a success event. This is a factual report, not a quality judgment (that would be Eval). |
| 0x0013 | Inform | Blocked | Report being blocked | (1,2,0,5) | The sender cannot continue its work due to a dependency or impediment. Negative polarity because this is a problem. Higher urgency because blockers impede progress and need resolution. |
| 0x0014 | Inform | Progress | Share progress update | (1,5,0,3) | The sender provides an incremental update on work that is underway but not yet done. Slightly positive polarity because progress is being made. Used for mid-task check-ins between start and completion. |
| 0x0020 | Ask | Clarify | Request clarification | (2,4,1,4) | The sender does not understand something and needs the receiver to explain or disambiguate. This is a question born from confusion or ambiguity, not a request for action. The desired response is information, not work. |
| 0x0021 | Ask | Status | Query current status | (2,4,0,3) | The sender wants to know the current state of a task, process, or system. The sender does not know the status and is asking the responsible party to report it. The expected response is Inform Status. |
| 0x0022 | Ask | Permission | Request permission | (2,4,4,4) | The sender wants to do something but needs authorization from the receiver first. This is "may I?" not "please do" -- the sender intends to act, but needs a green light. The expected response is Accept or Reject. |
| 0x0023 | Ask | Resource | Query resource availability | (2,4,5,3) | The sender wants to know whether a resource (compute, infrastructure, personnel) is available. This is an information request, not a resource allocation request (that would be Request Resource). |
| 0x0030 | Request | Task | Request task execution | (3,4,0,4) | The sender directs the receiver to perform a specific unit of work. This is the most common directive in a multi-agent system -- assigning work to another agent. The receiver is expected to either Accept and Commit, or Reject. |
| 0x0031 | Request | Plan | Request plan creation | (3,4,1,4) | The sender asks the receiver to create a plan or strategy. The receiver should produce a plan, not execute the work. Distinguished from Propose Plan, where the sender offers a plan rather than requesting one. |
| 0x0032 | Request | Review | Request work review | (3,4,3,3) | The sender asks the receiver to examine and evaluate a piece of work. The expected response is an Eval (Approve, NeedsWork, or Review). This is about soliciting evaluation, not about doing the work. |
| 0x0033 | Request | Help | Request assistance | (3,4,7,5) | The sender needs help with a problem and is asking the receiver for support. Higher urgency than a general task request because the sender is stuck. The domain is "general" because help requests can span any topic. |
| 0x0034 | Request | Cancel | Request cancellation | (3,1,4,5) | The sender directs the receiver to stop or cancel an ongoing or planned operation. Negative polarity because cancellation discards work. High urgency because time-sensitivity matters for cancellations. |
| 0x0035 | Request | Priority | Request priority change | (3,4,4,5) | The sender asks the receiver to change the priority or urgency of a task. High urgency because priority changes typically mean something has become more urgent. |
| 0x0036 | Request | Resource | Request resource allocation | (3,4,5,4) | The sender directs the receiver to allocate or provision a resource. Unlike Ask Resource (which queries availability), this requests that the allocation actually happen. |
| 0x0040 | Propose | Plan | Propose a plan | (4,5,1,4) | The sender offers a plan or strategy for consideration. The receiver may Accept, Reject, or counter-propose. This is a suggestion, not a directive -- the sender is offering an idea, not commanding action. |
| 0x0041 | Propose | Change | Propose modification | (4,5,0,4) | The sender suggests modifying something -- code, architecture, process, or configuration. This is softer than Request: the sender wants discussion before action. |
| 0x0042 | Propose | Alternative | Propose alternative | (4,5,1,4) | The sender suggests a different approach from what was previously discussed. This implies disagreement with the current direction but offers a constructive replacement rather than just rejecting. |
| 0x0043 | Propose | Rollback | Propose reverting | (4,3,4,5) | The sender suggests reverting to a previous state -- undoing a deployment, a code change, or a configuration update. Negative-leaning polarity because rollbacks imply something went wrong. Higher urgency because rollbacks are typically reactive. |
| 0x0050 | Commit | Task | Commit to task | (5,6,0,4) | The sender pledges to execute a task. This is a binding commitment -- the sender is taking ownership and will deliver. Positive polarity because commitment is a constructive act. |
| 0x0051 | Commit | Deadline | Commit to deadline | (5,6,0,4) | The sender pledges to deliver by a specific time. This binds the sender to a schedule. Often follows a Request Task or Accept, adding a time commitment. |
| 0x0052 | Commit | Resource | Commit resources | (5,6,5,4) | The sender pledges to allocate or provide resources. This is the sender committing their own resources, not requesting resources from others. |
| 0x0060 | Eval | Approve | Evaluation: approved | (6,7,3,4) | The sender has reviewed work and judges it as meeting standards. This is the highest polarity in the evaluation domain -- an explicit stamp of approval. Used after a Request Review flow. |
| 0x0061 | Eval | Review | Evaluation: under review | (6,4,3,4) | The sender indicates that evaluation is in progress but no verdict has been reached yet. Neutral polarity because the outcome is undetermined. Used as an intermediate state between Request Review and a final Eval verdict. |
| 0x0062 | Eval | NeedsWork | Evaluation: needs revision | (6,3,3,4) | The sender has reviewed work and judges that it requires changes before it can be approved. Negative-leaning polarity. This is not a rejection of the entire effort -- it is a constructive "not yet" with an expectation that the work will be revised. |
| 0x0063 | Eval | Complete | Evaluation: work complete | (6,6,3,4) | The sender confirms that reviewed work meets all criteria and is complete. Similar to Eval Approve but emphasizes completeness rather than quality approval. |
| 0x0070 | Meta | Ack | Acknowledge receipt | (7,5,4,2) | The sender confirms it received a prior message. This carries no commitment, agreement, or disagreement -- only the fact that the message arrived. Low urgency because acknowledgments are routine protocol. |
| 0x0071 | Meta | Sync | Synchronization ping | (7,4,4,3) | A liveness or synchronization check. The sender wants to confirm the receiver is alive and responsive. No semantic content beyond "are you there?" |
| 0x0072 | Meta | Handoff | Hand off responsibility | (7,4,4,4) | The sender transfers ownership or responsibility for a task or domain to the receiver. After this message, the receiver is the responsible party. |
| 0x0073 | Meta | Escalate | Escalate to authority | (7,3,4,6) | The sender raises an issue to a higher authority because it exceeds the sender's scope, capability, or authorization. High urgency because escalation implies the current level cannot resolve the issue. |
| 0x0074 | Meta | Abort | Abort operation | (7,0,4,7) | Emergency termination of an operation. The lowest polarity and highest urgency in the system. Used when something has gone critically wrong and all work must stop immediately. |
| 0x0080 | Accept | Generic | Accept proposal/request | (5,7,7,3) | The sender agrees to what was proposed or requested. The highest polarity in the system (7) because acceptance is a maximally positive response. Uses Generic because Accept alone carries the semantic weight. |
| 0x0081 | Reject | Generic | Reject proposal/request | (5,0,7,3) | The sender refuses what was proposed or requested. The lowest polarity for a non-emergency action (0). Uses Generic because Reject alone carries the semantic weight. |
| 0x0082 | Accept | Condition | Conditional acceptance | (5,5,7,4) | The sender agrees, but with stipulations. Lower polarity than Accept Generic because the acceptance is qualified. The condition should be specified in the payload. |
| 0x0083 | Meta | Defer | Defer decision | (5,4,7,2) | The sender postpones making a decision. Neutral polarity -- neither accepting nor rejecting, just delaying. Low urgency because deferral explicitly reduces time pressure. |
| 0x0090 | Error | Generic | Generic error | (1,1,6,5) | An error occurred that does not fit a more specific category. Negative polarity, error domain, elevated urgency. |
| 0x0091 | Error | Timeout | Operation timed out | (1,1,6,5) | An operation failed because it exceeded its time limit. Indicates a performance or availability problem rather than a logic error. |
| 0x0092 | Error | Resource | Resource unavailable | (1,1,6,5) | A required resource (compute, storage, external service) could not be obtained. Indicates a capacity or availability problem. |
| 0x0093 | Error | Permission | Permission denied | (1,0,6,5) | An operation failed because the sender or actor lacks the required authorization. The lowest polarity among errors because permission denial implies a security boundary violation. |
| 0x0094 | Error | Validation | Validation failed | (1,1,6,4) | Input, configuration, or data failed a validation check. Slightly lower urgency than other errors because validation failures are typically caught early and are recoverable. |
| 0x00FF | Fallback | Generic | Unquantizable - see ref | (7,4,7,4) | The content could not be mapped to any Force-Object pair with sufficient confidence. A pointer reference in the payload allows retrieval of the original text. This is a last resort, not a normal classification. |

---

## 4. Valid vs Invalid Combinations

### 4.1 The 45 Valid Core Combinations

These are the only Force-Object pairs defined in the core UCR. Each represents a specific, well-defined semantic intent.

| # | Force | Object | Summary |
|---|-------|--------|---------|
| 1 | Observe | State | Report what is seen right now |
| 2 | Observe | Change | Report a detected transition |
| 3 | Observe | Error | Report a witnessed fault |
| 4 | Inform | Result | Deliver computed output |
| 5 | Inform | Status | Self-report on work state |
| 6 | Inform | Complete | Announce task finished |
| 7 | Inform | Blocked | Report an impediment |
| 8 | Inform | Progress | Share incremental update |
| 9 | Ask | Clarify | Seek disambiguation |
| 10 | Ask | Status | Query work/system state |
| 11 | Ask | Permission | Seek authorization |
| 12 | Ask | Resource | Query availability |
| 13 | Request | Task | Assign work |
| 14 | Request | Plan | Ask for a strategy |
| 15 | Request | Review | Solicit evaluation |
| 16 | Request | Help | Ask for assistance |
| 17 | Request | Cancel | Direct a cancellation |
| 18 | Request | Priority | Direct a priority change |
| 19 | Request | Resource | Direct a resource allocation |
| 20 | Propose | Plan | Offer a strategy |
| 21 | Propose | Change | Suggest a modification |
| 22 | Propose | Alternative | Suggest a different approach |
| 23 | Propose | Rollback | Suggest reverting |
| 24 | Commit | Task | Pledge to do work |
| 25 | Commit | Deadline | Pledge to a timeline |
| 26 | Commit | Resource | Pledge resources |
| 27 | Eval | Approve | Judge work as passing |
| 28 | Eval | Review | Indicate evaluation in progress |
| 29 | Eval | NeedsWork | Judge work as needing revision |
| 30 | Eval | Complete | Confirm work meets criteria |
| 31 | Meta | Ack | Confirm message receipt |
| 32 | Meta | Sync | Liveness/sync check |
| 33 | Meta | Handoff | Transfer responsibility |
| 34 | Meta | Escalate | Raise to higher authority |
| 35 | Meta | Abort | Emergency termination |
| 36 | Accept | Generic | Agree to proposal/request |
| 37 | Reject | Generic | Refuse proposal/request |
| 38 | Accept | Condition | Agree with stipulations |
| 39 | Meta | Defer | Postpone a decision |
| 40 | Error | Generic | Unclassified technical error |
| 41 | Error | Timeout | Time limit exceeded |
| 42 | Error | Resource | Resource not available |
| 43 | Error | Permission | Authorization failure |
| 44 | Error | Validation | Data/input check failure |
| 45 | Fallback | Generic | Cannot quantize, see pointer |

### 4.2 Common Invalid Combinations and Why They Are Wrong

The following Force-Object pairs do NOT exist in the core UCR. For each, the explanation describes why the combination is semantically incoherent or redundant, and what the correct anchor should be instead.

**Observe + Task**: Observe is a passive perceptual act -- the sender reports what it sees. A task is a unit of work to be performed. You cannot passively observe "a task" in the abstract. If the sender noticed that a task was assigned, the observation is about a Change. If the sender is reporting task status, that is Inform Status. Correct alternatives: Observe Change (if a task assignment was detected), Inform Status (if reporting on task state).

**Observe + Plan**: Plans are not passively observed -- they are created, proposed, or requested. If the sender discovered that a plan exists, the relevant observation is Observe State (the plan is part of system state) or Observe Change (the plan was modified). Correct alternatives: Observe State, Observe Change.

**Observe + Complete**: If the sender noticed that something finished, it is observing a state change. The anchor is Observe Change (something transitioned to a completed state) or Inform Complete (if the sender's own work finished). Correct alternatives: Observe Change, Inform Complete.

**Inform + Task**: Inform reports information. Task is work to be done. You inform about a result, status, or completion -- not about "a task" directly. If the sender is reporting that a task was assigned, use Inform Status. If reporting that a task was completed, use Inform Complete. Correct alternatives: Inform Status, Inform Complete, Inform Result.

**Inform + Clarify**: Clarify is the object for disambiguation requests. Informing "a clarification" is semantically confused. If the sender is providing a clarification that was requested, the response is Inform Result (the clarification is a result of the Ask). Correct alternative: Inform Result.

**Inform + Permission**: You inform about results, status, and completion -- not about permissions. If granting permission, use Accept Generic or Accept Condition. If reporting that permission was denied, use Error Permission. Correct alternatives: Accept Generic, Error Permission.

**Ask + Task**: Ask is for questions. If the sender wants work done, use Request Task. "Asking someone to do a task" is a request, not a question. The dividing line: if the desired response is an action, use Request. If the desired response is information, use Ask. Correct alternative: Request Task.

**Ask + Complete**: "Asking a completion" is incoherent. If the sender wants to know whether something is done, use Ask Status (querying the current state). Correct alternative: Ask Status.

**Ask + Result**: If the sender wants to know the result of something, it is querying status. Use Ask Status. "Result" as an object is for the output being delivered, not for the query asking about it. Correct alternative: Ask Status.

**Request + State**: You cannot request "a state." If the sender wants to know the state, use Ask Status. If the sender wants to change the state, use Request Task (with the desired change in the payload). Correct alternatives: Ask Status, Request Task.

**Request + Result**: You cannot request "a result" as an object -- you request work (Task) that produces a result. The result is what comes back as Inform Result. Correct alternative: Request Task.

**Request + Complete**: You cannot request "a completion." You request a task, and when it is done, the receiver sends Inform Complete. Correct alternative: Request Task.

**Propose + Task**: Propose is for suggestions. If the sender is suggesting work be done, the sender is really proposing a plan (Propose Plan) or proposing a change (Propose Change). If the sender wants to assign work, use Request Task. Correct alternatives: Propose Plan, Propose Change.

**Propose + Status**: You cannot propose "a status." Status is something reported, not suggested. Correct alternative: Inform Status (if reporting), Ask Status (if querying).

**Commit + Review**: You commit to doing work, not to "a review." If the sender is pledging to review something, use Commit Task (the task is the review). Correct alternative: Commit Task.

**Commit + Approve**: You cannot commit "an approval." Approval is a judgment. If the sender approved something, use Eval Approve. Correct alternative: Eval Approve.

**Eval + Task**: Eval is for quality judgments. "Evaluating a task" is vague. If the sender approved the task output, use Eval Approve. If it needs revision, use Eval NeedsWork. If the evaluation is in progress, use Eval Review. Correct alternatives: Eval Approve, Eval NeedsWork, Eval Review.

**Eval + Status**: You do not evaluate "a status." If the sender is evaluating work and providing a status on that evaluation, use Eval Review (evaluation in progress). Correct alternative: Eval Review.

**Accept + Task**: Accept is a response to a proposal or request. You do not accept "a task" -- you accept a proposal (Accept Generic) or commit to a task (Commit Task). If the sender is saying "yes, I will do that task," the correct decomposition is Accept Generic (agreeing) followed by Commit Task (pledging to do the work). Correct alternatives: Accept Generic, Commit Task.

**Reject + Task**: Similar to Accept + Task. You reject a proposal or request (Reject Generic), not "a task." Correct alternative: Reject Generic.

**Error + Task**: Error reports technical failures. A task is not an error. If a task failed, use Error Generic (with context in payload). If execution of a task produced an error, use Error Generic, Error Timeout, Error Validation, or whichever specific error category fits. Correct alternatives: Error Generic, Error Timeout, Error Validation.

**Fallback + (anything other than Generic)**: Fallback exists specifically because the content could not be classified. Pairing Fallback with a specific object contradicts its purpose. If the object can be identified, then the content can at least partially be quantized and Fallback is not the right Force. Correct alternative: Fallback Generic (always).

### 4.3 Borderline Cases

These are situations where the correct classification is genuinely ambiguous. Training data should reflect this ambiguity by marking these as lower-confidence examples.

**"I noticed the tests passed" -- Observe Change vs Inform Complete**: If the sender ran the tests (is a participant), this is Inform Complete. If the sender is a monitoring agent that watches CI output (is an observer), this is Observe Change. The dividing line is whether the sender is a participant or a bystander.

**"Could you review this?" -- Ask vs Request**: "Could you" can be either a polite request or a genuine question about capability. In agent-to-agent communication, this is almost always Request Review. The key test: does the sender want information ("are you able to review?") or action ("please review")? In practice, treat this as Request Review unless the context strongly indicates the sender genuinely questions the receiver's ability.

**"I think we should use Redis" -- Propose Plan vs Propose Change vs Propose Alternative**: All three are plausible. Use Propose Alternative if this contradicts a previous plan (replacing one approach with another). Use Propose Change if modifying an existing design. Use Propose Plan if no prior plan exists and this is the first strategy offered. If there is no conversational context to distinguish, default to Propose Plan.

**"OK" -- Accept Generic vs Meta Ack**: "OK" after a proposal is Accept Generic (agreeing to proceed). "OK" after an Inform message is Meta Ack (acknowledging receipt). The meaning depends entirely on what "OK" is responding to. If responding to a Request or Propose, it is Accept Generic. If responding to an Inform, it is Meta Ack.

**"The migration script failed" -- Inform vs Error vs Observe**: If the sender ran the migration, this is Error Generic (the sender's operation failed). If the sender is a monitoring agent, this is Observe Error (the sender witnessed a failure). If the sender is reporting that someone else's migration failed as a status update, this is Inform Status. The key: who ran the migration, and what is the sender's relationship to it?

**"Stop everything" -- Request Cancel vs Meta Abort**: Meta Abort is for emergencies and protocol-level halts. Request Cancel is for normal workflow cancellation. The dividing line is severity: if the system is in danger, use Meta Abort. If the sender simply changed their mind or priorities shifted, use Request Cancel.

**"We need more memory" -- Ask Resource vs Request Resource**: Ask Resource queries whether memory is available ("do we have it?"). Request Resource directs allocation ("give us more"). If the sender is investigating, use Ask Resource. If the sender has decided and wants action, use Request Resource.

**"The code looks decent but has a few issues" -- Eval Approve vs Eval NeedsWork**: If the issues are blocking and require changes before merge, this is Eval NeedsWork. If the issues are minor and the work is acceptable as-is (perhaps with follow-up), this is Eval Approve. The test: does the sender expect revisions before proceeding?

**"I can take that" -- Accept Generic vs Commit Task**: If responding to a Request Task, this is Accept Generic (agreeing to the request) which should be followed by Commit Task. If the sender is proactively volunteering without being asked, this is Commit Task. In practice, a single Accept Generic can imply the commitment, but a separate Commit Task is more precise.

**"Let's revisit this later" -- Meta Defer vs Reject Generic**: Meta Defer postpones without prejudice -- the decision is delayed, not denied. Reject Generic is a definitive no. If the sender intends to reconsider in the future, use Meta Defer. If the sender is using "later" as a soft no, use Reject Generic.

---

## 5. DPO Preference Examples

For each Force, the following provides preference pairs for training. Each pair has a natural language input, the correct (chosen) Force-Object, and a plausible but incorrect (rejected) Force-Object with explanation.

### 5.1 Observe

**Example 1**
- Input: "The CPU usage on prod-3 jumped to 95%"
- Chosen: **Observe State** -- The sender is reporting a monitored metric. This is a passive observation of current system state.
- Rejected: **Inform Status** -- Inform is for self-reporting on the sender's own work. The sender is not working on CPU usage; it is watching a system metric. Observe is correct for external monitoring.

**Example 2**
- Input: "I detected that the config file was modified at 3:42 AM"
- Chosen: **Observe Change** -- The sender noticed a transition (file modification). This is a change detection, not a current-state snapshot.
- Rejected: **Observe State** -- State is for current snapshots, not transitions. The key word "modified" signals a change occurred, making Change the correct object.

**Example 3**
- Input: "The payment service is throwing 500 errors"
- Chosen: **Observe Error** -- The sender is witnessing errors in a system it monitors. It is not the payment service itself.
- Rejected: **Error Generic** -- Error (as a Force) is for when the sender's own operation failed. Here the sender is a third party observing another system's failures.

**Example 4**
- Input: "I see the deployment pipeline is currently running"
- Chosen: **Observe State** -- The sender reports the current state of the pipeline. It is a real-time observation, not a change.
- Rejected: **Inform Progress** -- Inform is for the sender's own work. The sender is not running the pipeline; it is observing it. Observe State is correct.

**Example 5**
- Input: "The new replica set just came online"
- Chosen: **Observe Change** -- The sender detected a state transition (replica came online). This is a change event.
- Rejected: **Inform Complete** -- The sender did not bring the replica online. Inform Complete is for announcing one's own work finishing. The sender is observing another system's event.

---

### 5.2 Inform

**Example 1**
- Input: "The API response time test shows p99 at 230ms"
- Chosen: **Inform Result** -- The sender produced or ran a test and is delivering the output. This is active sharing of a computed result.
- Rejected: **Observe State** -- The sender ran the test; it is not passively observing. Inform is correct because the sender is the actor delivering their own output.

**Example 2**
- Input: "I finished migrating the user table"
- Chosen: **Inform Complete** -- The sender is announcing that its own task is done.
- Rejected: **Eval Complete** -- Eval Complete is a quality judgment confirming work meets criteria after a review. The sender is not evaluating; it is reporting completion of its own work.

**Example 3**
- Input: "I am blocked waiting for the SSL certificate"
- Chosen: **Inform Blocked** -- The sender reports it cannot continue due to a dependency.
- Rejected: **Request Resource** -- While the sender might implicitly want the certificate, the direct speech act is reporting a blocker, not directing someone to provide a resource. The request is implicit, not the primary intent.

**Example 4**
- Input: "FYI, the staging environment is using 90% disk"
- Chosen: **Inform Status** -- The sender is sharing a status update proactively. "FYI" signals informational intent.
- Rejected: **Observe State** -- This is borderline, but "FYI" indicates the sender is actively sharing information (Inform), not passively reporting a monitored observation. If this came from an automated monitoring agent, Observe State would be correct.

**Example 5**
- Input: "3 of 7 migration steps are complete"
- Chosen: **Inform Progress** -- The sender shares an incremental update on ongoing work.
- Rejected: **Inform Complete** -- The work is not done (3 of 7). Progress is for partial completion updates. Complete is only for when the work is fully finished.

---

### 5.3 Ask

**Example 1**
- Input: "What do you mean by 'optimize the query'?"
- Chosen: **Ask Clarify** -- The sender does not understand and needs disambiguation.
- Rejected: **Request Help** -- The sender is not asking for help with a problem; it is asking for explanation of an ambiguous instruction. The desired response is information (clarification), not assistance with work.

**Example 2**
- Input: "Can I deploy to production?"
- Chosen: **Ask Permission** -- The sender wants authorization to act. The sender knows how to deploy; it needs approval.
- Rejected: **Request Task** -- The sender is not asking someone else to deploy. It is asking whether it is allowed to deploy. The desired response is a yes/no (Accept or Reject), not task execution.

**Example 3**
- Input: "Is the GPU cluster free right now?"
- Chosen: **Ask Resource** -- The sender queries resource availability. It wants information about capacity, not allocation.
- Rejected: **Request Resource** -- The sender is not (yet) asking for resources to be allocated. It is asking whether they are available. The distinction: Ask Resource is a question; Request Resource is a directive.

**Example 4**
- Input: "How far along is the database migration?"
- Chosen: **Ask Status** -- The sender wants to know the current state of work. It is querying for a status update.
- Rejected: **Inform Progress** -- Inform is for the sender to share information, not to request it. Ask is correct because the sender is the one lacking information.

**Example 5**
- Input: "Which version of the API should we target?"
- Chosen: **Ask Clarify** -- The sender needs disambiguation on a decision that affects its work.
- Rejected: **Ask Status** -- The sender is not asking about the status of ongoing work. It is asking for a decision or specification to be clarified.

---

### 5.4 Request

**Example 1**
- Input: "Please implement rate limiting on the /api/users endpoint"
- Chosen: **Request Task** -- The sender directs the receiver to perform a specific piece of work.
- Rejected: **Propose Change** -- The sender is not suggesting; it is directing. "Please implement" is imperative, not suggestive. The sender expects action, not discussion.

**Example 2**
- Input: "Can you look at my pull request for the auth module?"
- Chosen: **Request Review** -- Despite the question form ("can you"), the intent is to solicit a review. The desired response is an evaluation, not an answer to a yes/no question.
- Rejected: **Ask Clarify** -- The sender is not confused about something. "Can you look at" is a polite request for action (review), not a question seeking information.

**Example 3**
- Input: "I need help debugging the memory leak in the worker service"
- Chosen: **Request Help** -- The sender is stuck and asking for assistance.
- Rejected: **Inform Blocked** -- While the sender is stuck, the primary speech act is requesting help, not reporting a blocker. The emphasis is on "I need help" (directive), not "I am stuck" (informative).

**Example 4**
- Input: "Stop the deployment, we found a critical bug"
- Chosen: **Request Cancel** -- The sender directs cancellation of an ongoing operation.
- Rejected: **Meta Abort** -- Abort is for emergency protocol-level halts. This is a directed cancellation of a specific deployment. If the sender said "halt all operations immediately," Abort would be correct. "Stop the deployment" is scoped and directed.

**Example 5**
- Input: "We need 8 more GPUs for the training run"
- Chosen: **Request Resource** -- The sender directs allocation of resources.
- Rejected: **Ask Resource** -- "We need" is a demand, not a question. The sender is not asking whether GPUs are available; it is requesting that they be allocated.

---

### 5.5 Propose

**Example 1**
- Input: "I think we should switch from REST to gRPC for the internal services"
- Chosen: **Propose Alternative** -- The sender suggests replacing the current approach (REST) with a different one (gRPC).
- Rejected: **Propose Change** -- Change is for modifications to an existing design. Alternative is for replacing one approach with a fundamentally different one. Switching protocols is a replacement, not a modification.

**Example 2**
- Input: "How about we add a caching layer between the API and the database?"
- Chosen: **Propose Change** -- The sender suggests modifying the existing architecture by adding a component.
- Rejected: **Request Task** -- "How about" signals a suggestion, not a directive. The sender is opening discussion, not assigning work.

**Example 3**
- Input: "I suggest we roll back the last release; it introduced the regression"
- Chosen: **Propose Rollback** -- The sender suggests reverting to a prior state.
- Rejected: **Request Cancel** -- Cancel stops something in progress. Rollback undoes something already done. The release already happened; the suggestion is to revert it.

**Example 4**
- Input: "My recommendation is to split this into three microservices with separate databases"
- Chosen: **Propose Plan** -- The sender offers a strategy for consideration. This is a plan for how to proceed.
- Rejected: **Request Plan** -- Request Plan asks someone else to create a plan. Here the sender is providing a plan. The direction of the plan (from sender vs requested from receiver) determines the Force.

**Example 5**
- Input: "Instead of polling, we could use WebSockets for real-time updates"
- Chosen: **Propose Alternative** -- "Instead of" explicitly signals replacing one approach with another.
- Rejected: **Propose Plan** -- This is not a comprehensive plan; it is a targeted substitution of one technology for another. Alternative is more precise than Plan here.

---

### 5.6 Commit

**Example 1**
- Input: "I will handle the database migration this sprint"
- Chosen: **Commit Task** -- The sender pledges to do the work.
- Rejected: **Accept Generic** -- Accept is a response to a prior request or proposal. If nobody asked the sender to do this, it is a proactive commitment, not an acceptance. Even if someone did ask, "I will handle" emphasizes the commitment more than the agreement.

**Example 2**
- Input: "I can deliver the API documentation by next Friday"
- Chosen: **Commit Deadline** -- The sender pledges to a specific timeline.
- Rejected: **Commit Task** -- While a task is implied, the emphasis is on the deadline ("by next Friday"). When the time commitment is the salient information, Deadline is more precise than Task.

**Example 3**
- Input: "Allocating 16 GB of additional RAM to the staging cluster"
- Chosen: **Commit Resource** -- The sender is pledging their own resources.
- Rejected: **Request Resource** -- "Allocating" is an active commitment by the sender, not a request to someone else. The sender controls the resources and is committing them.

**Example 4**
- Input: "On it, I am taking this ticket"
- Chosen: **Commit Task** -- The sender is taking ownership of work.
- Rejected: **Meta Ack** -- "On it" carries more than acknowledgment. It is a commitment to act, not just a receipt confirmation. Ack says "I heard you." Commit Task says "I will do it."

**Example 5**
- Input: "I am assigning two of my team members to the security audit"
- Chosen: **Commit Resource** -- The sender is committing personnel (a resource) to a project.
- Rejected: **Request Task** -- The sender is not asking someone else to do something. The sender is allocating resources from their own team.

---

### 5.7 Eval

**Example 1**
- Input: "Looks good to me, approved for merge"
- Chosen: **Eval Approve** -- The sender has reviewed and renders a positive verdict.
- Rejected: **Accept Generic** -- Accept is for agreeing to proposals or requests. Eval Approve is for quality judgments after reviewing work. The sender is not agreeing to a proposal; it is judging submitted code.

**Example 2**
- Input: "The error handling needs improvement before we can merge this"
- Chosen: **Eval NeedsWork** -- The sender has reviewed and judges that revisions are required.
- Rejected: **Reject Generic** -- Reject is a definitive refusal. NeedsWork is a constructive "not yet" -- the sender expects the work to be revised and resubmitted. The intent is improvement, not refusal.

**Example 3**
- Input: "I am still reviewing the pull request, will have comments by EOD"
- Chosen: **Eval Review** -- The evaluation is in progress. No verdict yet.
- Rejected: **Inform Progress** -- While there is a progress element, the primary context is an ongoing evaluation. Eval Review specifically signals that a review is underway.

**Example 4**
- Input: "All acceptance criteria met, the feature is complete"
- Chosen: **Eval Complete** -- The sender confirms work meets all defined criteria.
- Rejected: **Inform Complete** -- Inform Complete is for the person who did the work to announce it is done. Eval Complete is for the reviewer confirming the work meets standards. The distinction: who is speaking? The worker informs; the reviewer evaluates.

**Example 5**
- Input: "Ship it"
- Chosen: **Eval Approve** -- "Ship it" is reviewer shorthand for approval.
- Rejected: **Request Task** -- The sender is not requesting that someone ship something. "Ship it" in a code review context is an approval verdict, not a deployment command.

---

### 5.8 Meta

**Example 1**
- Input: "Got it, I received the task description"
- Chosen: **Meta Ack** -- Pure receipt confirmation. No commitment or agreement is implied.
- Rejected: **Accept Generic** -- "Got it" in response to receiving information is acknowledgment, not agreement. Accept would mean "yes, I agree to do this." Ack means "yes, I received the message."

**Example 2**
- Input: "Ping, are you still there?"
- Chosen: **Meta Sync** -- A liveness check with no semantic content beyond synchronization.
- Rejected: **Ask Status** -- Ask Status queries the status of work. Meta Sync checks whether the other agent is responsive. The sender does not care about task status; it cares about connectivity.

**Example 3**
- Input: "Passing the authentication module ownership to the security team"
- Chosen: **Meta Handoff** -- Transfer of responsibility.
- Rejected: **Request Task** -- The sender is not assigning a new task. It is transferring ongoing responsibility for an existing domain. Handoff is about ownership transfer, not task assignment.

**Example 4**
- Input: "This decision requires VP approval, I am escalating"
- Chosen: **Meta Escalate** -- The sender raises an issue beyond their authority.
- Rejected: **Request Help** -- The sender is not asking for help with a problem. It is routing a decision to a higher authority because it exceeds the sender's scope. Help implies the sender will continue working with support; Escalate implies the sender is handing the decision upward.

**Example 5**
- Input: "Emergency: halt all production deployments now"
- Chosen: **Meta Abort** -- Emergency termination of operations.
- Rejected: **Request Cancel** -- Cancel is for normal-priority stoppage of a specific operation. Abort is for emergency-priority halting of everything. The word "emergency" and "halt all" signal maximum urgency.

---

### 5.9 Accept

**Example 1**
- Input: "Yes, let's go with that plan"
- Chosen: **Accept Generic** -- The sender agrees to a proposed plan.
- Rejected: **Commit Task** -- The sender is agreeing, not committing to do the work. Accept says "yes, I agree." Commit says "I will do it." These are different speech acts even if they sometimes co-occur.

**Example 2**
- Input: "Agreed, but only if we add comprehensive logging first"
- Chosen: **Accept Condition** -- The sender agrees with a stipulation.
- Rejected: **Accept Generic** -- Generic acceptance is unconditional. The "but only if" clause makes this conditional. Accept Condition is specifically for qualified agreement.

**Example 3**
- Input: "Confirmed, go ahead with the migration"
- Chosen: **Accept Generic** -- The sender gives unconditional agreement to proceed.
- Rejected: **Meta Ack** -- "Confirmed" here is not just acknowledging receipt. It is authorizing action ("go ahead"). Ack carries no authorization; Accept does.

**Example 4**
- Input: "Yes, provided that the rollback procedure is documented"
- Chosen: **Accept Condition** -- Conditional acceptance with a specific requirement.
- Rejected: **Propose Plan** -- The sender is not proposing a new plan. It is responding to an existing proposal with qualified agreement.

**Example 5**
- Input: "Affirmative"
- Chosen: **Accept Generic** -- Unconditional agreement.
- Rejected: **Meta Ack** -- "Affirmative" in response to a proposal or request means agreement, not just receipt. Context matters: if responding to a directive, this is Accept. If responding to an informational message, this would be Meta Ack. Default interpretation favors Accept.

---

### 5.10 Reject

**Example 1**
- Input: "No, that approach will not scale"
- Chosen: **Reject Generic** -- The sender refuses the proposed approach.
- Rejected: **Eval NeedsWork** -- NeedsWork means "revise and resubmit." Reject means "no, this approach is wrong." The sender is not asking for improvements; it is refusing the direction entirely.

**Example 2**
- Input: "I disagree, we should not use a NoSQL database for this"
- Chosen: **Reject Generic** -- The sender explicitly disagrees with a proposal.
- Rejected: **Propose Alternative** -- While the sender might later propose an alternative, the primary speech act here is rejection. The sentence disagrees and says what should NOT happen. If it also said what should happen instead, Propose Alternative would be appropriate as a follow-up.

**Example 3**
- Input: "Declined, the risk is too high"
- Chosen: **Reject Generic** -- Explicit refusal with reasoning.
- Rejected: **Meta Defer** -- Defer means "not now, maybe later." Reject means "no." The sender is not postponing; it is refusing based on risk assessment.

**Example 4**
- Input: "That request is outside the scope of this sprint"
- Chosen: **Reject Generic** -- The sender refuses a request on scoping grounds.
- Rejected: **Meta Defer** -- This could be borderline. If the sender means "not this sprint, but maybe next sprint," it could be Defer. But "outside the scope" more strongly implies rejection rather than postponement. If the sender intended to revisit, they would typically say so explicitly.

**Example 5**
- Input: "I refuse to deploy without passing integration tests"
- Chosen: **Reject Generic** -- The sender refuses a specific action.
- Rejected: **Inform Blocked** -- The sender is not reporting a blocker. It is actively refusing to proceed. The distinction: Blocked is involuntary ("I cannot"), Reject is voluntary ("I will not").

---

### 5.11 Error

**Example 1**
- Input: "Database connection failed: connection refused on port 5432"
- Chosen: **Error Generic** -- A technical failure in the sender's operation.
- Rejected: **Observe Error** -- Observe Error is for a monitoring agent witnessing another system's failure. If the sender was trying to connect and failed, it is Error (the sender's own operation failed). If the sender is a monitoring agent that noticed the database is down, it is Observe Error.

**Example 2**
- Input: "The API request to the payment provider timed out after 30 seconds"
- Chosen: **Error Timeout** -- A specific timeout failure.
- Rejected: **Error Generic** -- While Generic is technically not wrong, Timeout is more precise. When a specific error subcategory exists and matches, always prefer the specific over the generic.

**Example 3**
- Input: "Service account lacks permissions to write to the S3 bucket"
- Chosen: **Error Permission** -- An authorization failure.
- Rejected: **Error Validation** -- Validation is for data/input format errors. Permission is for authorization failures. The issue is not bad data; it is insufficient access rights.

**Example 4**
- Input: "The request body is missing the required 'email' field"
- Chosen: **Error Validation** -- Input data failed a schema check.
- Rejected: **Error Generic** -- Validation is the specific subcategory for data format and schema errors. Prefer the specific type.

**Example 5**
- Input: "No GPU instances available in the us-east-1 pool"
- Chosen: **Error Resource** -- A required resource could not be obtained.
- Rejected: **Inform Blocked** -- Error Resource is for technical resource unavailability. Inform Blocked is for workflow impediments. The distinction: Error Resource is a system-level failure ("the pool is empty"), while Inform Blocked is a workflow-level report ("I cannot continue"). If the sender is a resource manager reporting pool exhaustion, Error Resource is correct. If the sender is a developer saying "I need GPUs and cannot get them," Inform Blocked might be more appropriate.

---

### 5.12 Fallback

**Example 1**
- Input: "So I was thinking about the architecture yesterday and I have some thoughts on the caching layer but also I wanted to mention that the CI pipeline has been flaky and oh by the way did you see the email about the new compliance requirements?"
- Chosen: **Fallback Generic** -- Multi-topic, multi-intent message that cannot be cleanly mapped to a single Force-Object pair.
- Rejected: **Inform Status** -- This message contains at least three separate intents (architecture thoughts, CI observation, compliance question). Forcing it into a single anchor loses critical information. Fallback preserves the original text via pointer.

**Example 2**
- Input: "..."
- Chosen: **Fallback Generic** -- Empty or ambiguous content with no discernible intent.
- Rejected: **Meta Sync** -- Sync is an active liveness check. Ellipsis carries no clear communicative intent and does not match any Force-Object pair.

**Example 3**
- Input: "The report contains seventeen sections covering financial projections, market analysis, competitor landscape, product roadmap, engineering capacity, hiring plan, infrastructure costs, support staffing, partnership opportunities, regulatory compliance, risk assessment, timeline estimates, budget allocation, KPI definitions, success metrics, stakeholder communication plan, and executive summary."
- Chosen: **Fallback Generic** -- This is a content description, not a speech act. There is no clear Force (the sender is not observing, informing, asking, requesting, proposing, committing, evaluating, or performing any other act).
- Rejected: **Inform Result** -- While the sender is technically sharing information, the content is a raw data listing with no actionable intent. The quantizer should fall back when it cannot determine what the sender wants the receiver to do with this information.

---

## Appendix A: Coordinate System Reference

The four-dimensional coordinate system (ACTION, POLARITY, DOMAIN, URGENCY) maps each anchor to a point in semantic space. These coordinates enable nearest-neighbor matching when exact Force-Object resolution fails.

| Dimension | Index | Scale | Meaning |
|-----------|-------|-------|---------|
| ACTION | 0 | 0-7 | 0=observe, 1=inform, 2=ask, 3=request, 4=propose, 5=commit, 6=evaluate, 7=meta |
| POLARITY | 1 | 0-7 | 0=most negative, 4=neutral, 7=most positive |
| DOMAIN | 2 | 0-7 | 0=task, 1=plan, 2=observation, 3=evaluation, 4=control, 5=resource, 6=error, 7=general |
| URGENCY | 3 | 0-7 | 0=background, 4=normal, 7=critical |

## Appendix B: Decision Tree for Force Selection

Use this tree when classifying natural language into a Force token:

1. Is the sender reporting a technical failure in its own operation? --> **Error**
2. Is the sender's intent unclassifiable? --> **Fallback**
3. Is the sender responding to a prior proposal or request?
   - Affirmatively? --> **Accept**
   - Negatively? --> **Reject**
4. Is the sender performing protocol/coordination management (ack, sync, handoff, escalate, abort, defer)? --> **Meta**
5. Is the sender making a quality judgment about work? --> **Eval**
6. Is the sender pledging to do something or provide resources? --> **Commit**
7. Is the sender suggesting something for discussion? --> **Propose**
8. Is the sender directing someone to perform an action? --> **Request**
9. Is the sender seeking information, clarification, or permission? --> **Ask**
10. Is the sender reporting its own status, results, completion, progress, or blockers? --> **Inform**
11. Is the sender reporting something it passively observed in an external system? --> **Observe**

The order matters. Check from top to bottom and use the first match.

## Appendix C: Decision Tree for Object Selection

After determining the Force, use these heuristics for Object:

**For Observe**: Is it a current snapshot (State), a transition (Change), or a fault (Error)?

**For Inform**: Is it output data (Result), current work state (Status), finished (Complete), stuck (Blocked), or partial (Progress)?

**For Ask**: Is it about something unclear (Clarify), about current state of work (Status), about authorization (Permission), or about capacity (Resource)?

**For Request**: Is it for work (Task), a strategy (Plan), evaluation (Review), assistance (Help), stopping (Cancel), urgency change (Priority), or allocation (Resource)?

**For Propose**: Is it a strategy (Plan), a modification (Change), a replacement (Alternative), or a revert (Rollback)?

**For Commit**: Is it for work (Task), a timeline (Deadline), or allocation (Resource)?

**For Eval**: Is the verdict positive (Approve), in progress (Review), negative but constructive (NeedsWork), or confirming completeness (Complete)?

**For Meta**: Is it receipt confirmation (Ack), liveness (Sync), ownership transfer (Handoff), authority escalation (Escalate), emergency stop (Abort), or postponement (Defer)?

**For Accept**: Is it unconditional (Generic) or qualified (Condition)?

**For Reject**: Always Generic.

**For Error**: Is it unclassified (Generic), time-based (Timeout), capacity-based (Resource), authorization-based (Permission), or data-format-based (Validation)?

**For Fallback**: Always Generic.
