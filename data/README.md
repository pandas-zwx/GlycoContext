# Data construction, filtering, and coordinate validation

## What the data represent

The current dataset is a **site-by-cell proxy-classification cohort for human HeLa and H9 cells**, containing 11,479 records and 499 distinct database RNA sequences. One RNA may contain multiple candidate sites, and the same site may be represented separately in different cells. The number of records therefore does not equal the number of independent RNAs or independent experiments.

| Cell | Label 1 | Label 0 | Total |
|---|---:|---:|---:|
| HeLa | 1,910 | 3,025 | 4,935 |
| H9 | 2,439 | 4,105 | 6,544 |
| Total | 4,349 | 7,130 | 11,479 |

RNA sequences originate from the original human entries in GlycoRNAdb. Signals originate from the four cell-by-library samples in GSE136967 and their strand-specific tracks. Positive and negative genomic strands are not biological replicates, and HeLa and H9 are not replicate measurements of the same cell state. Mouse data are not included.

## Construction of positive and negative labels

For the **same cell, chromosome, strand, and genomic nucleotide**, the stored observations in the enrichment and input BigWig tracks are compared:

- Label 1: a stored observation is present in the enrichment library, with no corresponding stored observation in the input library.
- Label 0: a stored observation is present in the input library, with no corresponding stored observation in the enrichment library.
- Observations stored in both libraries are excluded from the binary cohort.
- When a sequence-position-cell candidate maps to multiple database entries or genomic sources, all sources must agree on the category; conflicting candidates are excluded.

Negatives are not sampled randomly from unreported positions. The absence of a stored observation is not interpreted as a measured biological zero, and no additional minimum read-count threshold is imposed. The input library is not a chemically validated negative control for the absence of glycosylation. Both y=0 and y=1 are therefore operational proxy labels. RT stops can also arise from other modifications, RNA structure, or natural termini; this cohort must not be described as site-resolved chemically validated ground truth.

## Reference sequences and coordinates

1. The fixed original human snapshot contains 589 entries and 524 distinct sequences. Entry aliases and different genomic sources for the same sequence are not counted automatically as independent RNAs.
2. GRCh38/GRCh38.p14 reference sequences and annotations are used to verify exact matches between RNA sequences and genomic or exon paths. A total of 573 entries and 512 distinct sequences have a usable unique validated path; 16 entries do not.
3. A genomic span is not required to equal the RNA length. Reference-validated multi-exon paths and interpretable coordinate representations are allowed, avoiding false sequence-error calls for intron-spanning RNAs or alternative coordinate conventions.
4. Positive-strand RNAs are mapped through exons in ascending order. Negative-strand RNAs use exons in descending order and map right-to-left within each interval. All final coordinates follow the RNA's own 5-prime-to-3-prime orientation.
5. Source BigWig coordinates are zero-based and half-open. `RNA_position1` is one-based in RNA orientation, and `rna_index0=RNA_position1-1`. Database position strings are not universally treated as BED coordinates, and no additional assumed RT offset is added or subtracted.
6. Every retained source is checked by RNA-to-genome-to-RNA round-trip mapping, and the center nucleotide must agree with the database RNA sequence. RNA windows are extracted only from that RNA sequence. Termini are padded with N and an independent `valid_mask`; genomic flanks or introns are never used to complete a window.

Context strings and structure-figure coordinates provided by the API are used as auxiliary validation evidence. Coordinates are not decided solely by 7-mer uniqueness, and an SVG is not required for every RNA. Candidate centers come from nucleotide-expanded source-track observations; bases in the surrounding API context are not all labeled as positive sites. Display orientation on the negative genomic strand must not be confused with RNA orientation: model inputs always use the database RNA sequence in the 5-prime-to-3-prime direction.

## Exclusions and reasons

| Item | Count | Unit and reason |
|---|---:|---|
| Stored observations in both libraries | 4,209 | Sequence-site-cell candidates that do not satisfy mutual exclusivity |
| Conflicting source categories | 162 | Sequence-site-cell candidates for which different sources disagree |
| API-unconfirmed candidates at genomic left boundaries | 8 | Present in source tracks but not covered by the API; conservatively isolated without claiming chemical false positivity |
| Entries without a usable reference path | 16 | Database entries; this count cannot be added directly to the candidate counts above |

After the eight boundary candidates are removed, 11,479 records remain. No additional strong-versus-weak signal threshold is introduced. Only 499 distinct sequences produce eligible candidates. Entries, sequences, and candidates are different statistical units and must not be subtracted as though they formed a single linear filtering cascade.

## Files and auditing

- `sites.jsonl.gz`: candidate ID, cell, RNA hash, zero- and one-based positions, center nucleotide, group, and proxy label.
- `sequences.jsonl.gz` and `sequences.fasta`: original database RNA sequences and their aliases, types, and sources; sequences are not inferred from RT-stop data.
- `provenance/source_observations.jsonl.gz`: source entry, strand, genomic coordinate, the two stored library observations, and track names for every candidate. This file is for auditing and **must not be used as a model input**.
- `provenance/reference_paths.json`: reference exon paths corresponding to each source, including evidence for multi-exon and negative-strand mappings.
- `provenance/exclusions.json`: exclusion counts and the eight boundary records.
- `provenance/sources.json`: public sources, versions, and source-snapshot digests.
- `manifest.json`: SHA256 digests for data files. Any data modification requires a new version and must not overwrite the original manifest.

```bash
python scripts/check_data.py --data data --splits splits
```

This check replays sequence and nucleotide consistency, source categories, coordinate round trips, and split integrity using the distributable files. It does not pretend to redownload the reference genome, rerun raw sequencing, or obtain chemical confirmation. The limitations of the proxy labels, API snapshot, and source libraries must be retained. Coordinate correctness must not be rewritten as evidence that a chemical modification has been proven.
