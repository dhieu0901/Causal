# Third-party notices

The project code, project-generated analysis tables, and the released response
snapshot are provided under the repository's MIT License to the extent that the
project authors hold rights in them. The following upstream materials retain
their own notices and licenses.

## CLadder

This project transforms prompts and metadata from CLadder, Copyright 2023
CausalNLP. CLadder is distributed under the MIT License:

https://github.com/causalNLP/cladder

https://github.com/causalNLP/cladder/blob/main/LICENSE

The response snapshot can contain short repetitions of CLadder-derived prompt
text inside model responses. The CLadder copyright and license notice therefore
apply to those portions.

## CaLM

This project analyzes locally downloaded material from CaLM. CaLM is
distributed under the Apache License 2.0:

https://github.com/OpenCausaLab/CaLM

https://github.com/OpenCausaLab/CaLM/blob/main/LICENSE

CaLM source data are not redistributed by this repository. Users download the
data from the upstream project and verify the expected checksums with
`python scripts/verify_data_provenance.py`.

## Model responses

`results/cladder/raw/analysis_response_snapshot.csv` contains the minimum
stored model-response text needed by four offline analyses. It contains no API
keys, account identifiers, request identifiers, timestamps, or billing data.
The release manifest is 5,239 rows and SHA-256
`01b1f96dc50a2371e027e25e48aaa70412a57bbda26f78ff82a66b0e1bc0f92a`.
The snapshot does not grant rights to upstream benchmark text beyond the
licenses stated above.
