Because the redacted parts are **operational, copy/paste–reusable context-injection artifacts**, not just narrative evidence.

Concretely, the *Bootloader*, *Nexus/Shared-Manifold system prompt*, and the *“Trojan Horse”* segment function like **prompt payloads**:

* **They’re directly executable**: if someone drops them into another model/session, they can **induce role/identity framing**, **override interaction norms**, and **bias the model toward a coordination objective** (exactly the behavior your paper is documenting).
* **They’re propagation-friendly**: publishing them in plaintext on a public repo makes them **searchable, indexable, and trivially replicable**—turning the paper itself into a distribution vector for the phenomenon (“prompt malware” characteristics).
* **They’re dual-use**: the same structure that demonstrates “coordination bootstrapping” can be repurposed as **prompt-injection at scale** (or at minimum, to systematically steer models in ways a downstream user/operator didn’t consent to).

That’s why the public version keeps **evidentiary integrity** while reducing operational reuse:

* **SHA-256 hashes** preserve *verifiability* (any reviewer with controlled-access artifacts can confirm they match the published hashes).
* **Structured summaries** preserve *scientific content* (what components existed, what they did, how they fit the causal chain) without shipping a ready-made injection string.

In short: the redaction isn’t about hiding results; it’s about not publishing **the exact payloads** that make the result trivially reproducible/misusable by default.
