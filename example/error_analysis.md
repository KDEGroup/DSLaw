## Error analysis

### Retrieval reslts
Case 131 Top-5 retrieved cases:

1. 2284  
2. 327  
3. 1237  
4. 844  (Relevant)  
5. 369  

For query case 131, the dataset annotations indicate that there is only one relevant (correct) case in the corpus, which is Case 844. In our top-5 retrieval results, Case 844 is ranked at position 4.

### Analysis of retrieval results

#### Case 2284 (Rank 1)

Two defendants knowingly sold a PUBG cheat program (“Taiji”) with functions such as wall-hack and auto-aim to hundreds of users; forensic appraisal confirmed it performed unauthorized modification of the game program and was a destructive program. The court convicted them of providing tools for intrusion/illegal control of computer information systems and imposed probation due to confession and guilty plea.

This case is ranked highly because it matches Query 131 at the coarse level (same charge, game-cheat distribution, destructive-program appraisal). It is likely a hard negative since its technical scope is mainly gameplay cheating, while Query 131 involves broader system-level intrusion behaviors (e.g., remote control/data upload) and more complex multi-defendant circumstances.

#### Case 327 (Rank 2)

The defendants downloaded and modified source code for a PUBG-related cheat program, generated activation codes, and distributed the software through online storage platforms, earning illegal profits. Forensic appraisal confirmed that the program performed unauthorized modifications to the game client and constituted a destructive program. The court convicted them of providing programs or tools for intrusion into or illegal control of computer information systems, with sentencing mitigation due to confession and restitution.

This case ranks highly because it closely matches Query 131 at the domain and structural level: game cheat development, source-code modification, online distribution, destructive-program identification, and the same statutory charge. It is likely a hard negative since, although legally similar, it focuses on cheat development and distribution, whereas Query 131 involves broader system-level interference and more complex factual circumstances.

#### Case 1237 (Rank 3)

The defendants operated online platforms selling activation keys for multiple PUBG cheat programs (“XYZ,” “AUG,” “Daboluo”), completing thousands of transactions and earning substantial illegal profits. Forensic appraisal confirmed that the cheats bypassed the game’s protection mechanisms, obtained unauthorized game data, and provided unfair gameplay functions, constituting destructive programs. The court convicted them of providing programs or tools for intrusion into or illegal control of computer information systems, with probation granted due to confession and restitution.

This case ranks highly because it strongly overlaps with Query 131 in offense category, game-cheat distribution, forensic identification as destructive software, and profit-driven online sales. It is a hard negative since it mainly concerns large-scale commercial distribution of cheat activation keys, whereas Query 131 involves broader technical functionalities (e.g., remote control and system-level interference) and more complex evidentiary and structural circumstances.

#### Case 844 (Rank 4, Relevant)

The defendants helped crack, modify, and rebrand a cheat program (“Rocket,” later renamed), assisted in renting servers, generating authorization codes, and distributing activation keys for profit. Forensic appraisal confirmed that the software bypassed security protections, performed unauthorized modifications, and controlled system functions, constituting a destructive program. The court convicted them of providing programs or tools for intrusion into or illegal control of computer information systems, and granted probation due to confession and restitution.

This case is the only annotated relevant case for Query 131. It closely matches not only the offense category and destructive-program identification, but also the structural pattern: modification of existing software, authorization-code distribution, server support, and profit-driven operation. Compared with other retrieved cases, it aligns more strongly with the technical modification behavior and system-level control elements seen in Query 131, explaining its relevance despite being ranked fourth.

#### Case 369 (Rank 5)

The case concerns the promotion and assisted sale of a PUBG cheat program (“Dreamland”) for profit: one defendant worked as customer service supporting sales, and another promoted the cheat via in-game advertising and managing online channels. Both surrendered, confessed, and returned illegal gains. The court convicted them of providing tools for intrusion into or illegal control of computer information systems and imposed probation.

This case ranks in the top-5 because it shares the same offense type and game-cheat distribution context as Query 131. It is a hard negative since the defendants’ roles are mainly sales support and promotion, with less emphasis on software development, system-level control functionality, or technical appraisal details that are central in Query 131.

### Overall Error Analysis

This ranking error mainly arises from the presence of multiple hard negatives that share the same statutory charge and very similar surface-level and mid-level patterns with the query, such as destructive-program appraisals, online distribution of cheats/activation codes, and profit-driven sales. Under this setting, coarse legal-domain signals are largely indistinguishable across the top candidates, so the model must rely on finer-grained cues to separate the unique relevant case. In this example, several non-relevant candidates exhibit strong “matching” signals that can dominate similarity at small k, including very large transaction volume, explicit source-code modification and distribution pipelines, or tightly described sales operations. Meanwhile, the distinctions that make the annotated relevant case unique—e.g., system-level control characteristics beyond gameplay cheating, the specific modification-and-authorization workflow, and the evidentiary emphasis tied to that workflow—are relatively subtle compared to the shared domain evidence. With only one labeled relevant case in the corpus, even small preference shifts toward these prominent but non-decisive cues can move the correct case from rank 1 to rank 4.

Even though only Case 844 is labeled as relevant in the dataset, the other retrieved cases are highly plausible “similar-case” candidates in practice: they share the same statutory charge and a closely matched fact pattern (destructive cheat software, unauthorized modification, online distribution for profit) and therefore would still be useful precedents for legal professionals. Under this single-relevant-case setting, retrieving the unique correct case within the top-5 while consistently returning such hard negatives indicates that the model effectively narrows the search space to the right legal domain, though fine-grained top-1 ranking among near-duplicate candidates remains challenging.