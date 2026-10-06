# Exact original Q4 recovery lookup

Read-only local lookup found the original HF BF16 snapshots/config pins and both
historical native source commits. The exact original Q4 outputs and their full
check/conversion receipts are absent from the inspected local artifact inventory.
No RTX2080Ti query, remote operation, build, conversion or quantization occurred.

Converter source:76847aa877261817026e2f29472f080235b6a829. Quantizer source:
fcdf5822c5b78f9dbcfd1f5c7106f09c3f0c9b1a. Preserved Darwin arm64 CPU quantizer:
`runs/dspark-sm75-local-preserved/native-cpu-build/bin/llama-quantize`, SHA
b73e5516f0f33993f26bfa64319c2d6f946f73f32d8c134e7129ae7d385a038f.
Its preservation receipt pins fcdf582; loader usability was not executed.

The historical report documents BF16 conversion with the original target
tokenizer, followed by pure Q4_0 for exactly the15 FFN gate/up/down matrices,
leaving output/token embeddings and other tensors BF16. The non-imatrix Q4
reference implementation is deterministic, but whole-file recovery additionally
requires identical original converter metadata and tensor ordering.

| Family | Current local BF16 SHA | Required original BF16 SHA |
|---|---|---|
| DSpark |3685ea7785729b4d3e9de3939264c1c87eb98a2bf7935a8c838eccf36be9fd13|dc5299bdb1e7e906b003334a9b806ba502e72111341432ac5a9fd11de22f520c|
| DFlash |61b294dc798c507fdc4a87f397b4015a78b3cdd7edad1e940e683bba10fb5575|92925d9e4be49a82d1e4f9d9671aae0e3647c1146fccbd4004bfc8afe0c9896f|

Both current containers have `general.name = Conversion Input`; the old names,
literal conversion argv, metadata overrides and input directory names are not
documented locally. The quantizer copies metadata. These current bases therefore
cannot be substituted as byte-identical originals. A bounded reconstruction is
only a hypothesis until the original BF16 hashes match, then original Q4 hashes
a95358608ce2e1c77f9a8253b692e697e3745dd68ba66b859833d41c480193fe and
6f23297bb623217e3df6352c8e83521f59f3b307b26f27f1262269775552b7c3 match.
Do not change control pins, relabel different bytes, or use the other GPU to
retrieve them. Original accessible files/complete serialization provenance remain
the decisive missing inputs. This does not admit final nine-model comparison.
