# Interview questions

1. **What does a schema guarantee?** A schema specifies acceptable instances.
   A generation system may enforce a supported subset. Local validation checks
   the application contract; neither mechanism establishes factual correctness.
   Completion and refusal handling remain separate from a valid response body.

2. **Why use both parsing and Pydantic?** Parsing interprets serialization;
   strict typed validation checks fields and values. Here the parser also rejects
   duplicate keys and non-finite constants before Pydantic sees a Python object.
   The exported schema is useful for an interface but does not carry arbitrary
   Python business rules or the reference oracle.

3. **How is missing different from null?** Required means the key must exist.
   Nullable means its value may explicitly be unknown. `purchase_order` is
   required and nullable; money fields in this narrow example require known
   values. A real extraction contract must define how partial records abstain.

4. **Why do integer cents still need validation?** Units and currency must be
   agreed, values bounded and cross-field identities checked. This example uses
   strict integers and no decimal conversion. Currency-specific rounding, credits,
   discounts and multi-rate taxes would require additional domain rules.

5. **Can constrained decoding prevent fabrication?** It restricts continuation
   probabilities, not source truth. Both `100` and `200` may be grammatical.
   Exact reference equality rejects a fabricated synthetic invoice here, but
   inference usually lacks a complete answer key.

6. **How would you select retries?** Classify failures first. The example retries
   parsing, schema and arithmetic errors within a total-call budget, stopping
   on reference mismatch, refusal, incomplete status and size violations. A real
   adapter needs distinct transport recovery, deadlines and cost accounting.
   Sensitive effects require authorization and idempotency independently.

7. **Why is the retry reliability formula potentially misleading?**
   $1-(1-p)^n$ assumes independent attempts with constant success probability.
   Ambiguous sources and repeated model errors correlate failures. Feedback
   changes the conditional recovery distribution, and passing validation may
   still accept a wrong record. This study measures none of those probabilities.

8. **What would you measure before accepting this design in a product?**
   Contract compliance, field accuracy, incorrect accepted records, missing-data
   coverage, terminal outcomes, attempts, cost and latency over all held-out
   requests. Also verify schema evolution, provider compatibility and execution
   permissions. The authored fixtures only establish selected software behavior.

9. **Is local masking equivalent to conditioning the model on a valid final
   output?** Generally no. Exact conditional sampling also depends on how likely
   each continuation is to reach a valid completion. A local mask excludes
   forbidden next tokens and renormalizes without that look-ahead weighting.

An interview explanation should connect the typed boundary to a concrete
failure: an invoice can parse, satisfy its fields and balance arithmetically,
yet contain a fabricated subtotal. Each acceptance check answers a different
question, and any downstream effect still requires its own execution authority.
