# VIDA 2014D schema map (empirically discovered, 2026-06-11)

Discovered by direct exploration of the attached databases (SQL Server 2022 in
Docker, container `vida-sql`), not by Profiler tracing — no Windows/VIDA client
was available. Every claim below was verified by query; blob patterns verified by
end-to-end decompression.

## Where the Information-tab documents live

**`servicerep_en-US.dbo.Document` — 86,517 rows, ALL with content blobs.**

| column | meaning |
|---|---|
| `id` (int, PK) | negative ints; our `source_pk` |
| `chronicleId` (char16) | Documentum id `0900…`; names the XML inside the blob |
| `projectDocumentId` (char16) | Documentum id `0800…`; join key for tree/links |
| `vccNumber` | the human doc ref, e.g. `VCC-140691-1` |
| `title` | document title |
| `fkDocumentType` | → `DocumentType.name` = XML root element |
| `fkQualifier` | → `Qualifier` → `QualifierGroup.name` (Repair / Parts / Diagnostic / Product specifications / Bulletins / Forms) |
| `XmlContent` (image) | **PKZIP containing exactly one file `<chronicleId>_en-US.xml`** |
| `path` | mirrors zip entry, e.g. `VCC-140691-1\0900c8af82829fa9_en-US.xml` |

Document counts by type: diag:servinfosub 51,799 · servinfosub 28,143 (repair/install
procedures) · diagnostic 4,180 · i2 1,446 · torqueinfo 663 · sbull 286.

XML body markup: `procedure/stepgrp/procstep/title/ptxt/list1/item/grate/graphic/xref`.
Root attrs: `docno` (VCC ref w/o revision), `IE-ID` (= `en-US`+chronicleId).
Images are inline refs: `<graphic id=…><href title="S2101307">0900c8af8005dd79_0_0.jpeg</href></graphic>`
— files live in the imagerepository DB / localgraphics store.
Torqueinfo example: `<torqueinfo …><component>M5</component><value unit="Nm">5</value></torqueinfo>`.

Linkage:
- `DocumentProfile` (2.14M rows): `fkDocument` → `profileId` (char16) — per-vehicle filter.
- `TreeItemDocument.projectDocumentTo = Document.projectDocumentId` → `TreeItem`
  (functionGroup1/2/3, tocLevel, title — the Information-tab tree), filtered by `TreeItemProfile`.
- `DocumentLink`/`DocumentLinkTitle`: cross-document hyperlinks (`en-US<chronicleId>#<element>`).
- `IndexedWord`/`DocumentIndexedWord`: VIDA's own FTS index (useful for validation).

## Vehicle profiles — `BaseData.dbo.VehicleProfile` (139,578 rows)

`Id` is a char(16) hex string — the universal cross-DB profile key. Attributes via
`fkVehicleModel/fkModelYear/fkEngine/fkTransmission/…` → lookup tables (`Id`,`Cid`,`Description`).
`FolderLevel` = how many attributes are set (the profile tree depth).

**Our car:** S60 (-09), 2005, B5254T4 → profile `0b00c8af815cadc9` (level 3).
Ancestors: `0b00c8af80da257a` (S60 2005, level 2), `0b00c8af8020665d` (S60, level 1).
Variants below it incl. `0b00c8af816362d7` (AW50/51 AWD), `0b00c8af816362fa` (M66 AWD),
market variants (AME `0b00c8af816678f6`…). Engine row: Id=1080, Cid=170, `B5254T4`;
ModelYear Id=1191 (2005); VehicleModel fk=1009 (S60).

## Other databases

- **DiagSwdlRepository**: `ScriptContent.XmlDataCompressed` (89,534 rows) = same
  ZIP-of-one-XML pattern, root `<script>` — diagnostic flow graphs, NOT phase-1 documents.
  `Language`: 15=en-US, 16=en-GB. The big `IE`/`SymptomIEMap` subsystem holds document
  *pointers* (titles + DTC/symptom/profile maps), no bodies.
- **CarCom**: text pool `T190_Text`/`T191_TextData` (en-US = fkT193_Language **19**,
  en-GB = 4), `T192_TextCategory`. ECU/diagnostic dictionary (T10x/T12x/T14x), own
  profile mirror `T161_Profile` (identifier = same char16 ids). NOT needed for
  servicerep extraction — document text is inline in the XML.
- **EPC**: pure relational parts catalog. `PartItems` (150,436 part numbers) →
  descriptions via local `Lexicon` (`DescriptionId`+`fkLanguage`, en-US=15).
  Catalogue tree `CatalogueComponents` (TypeId: 2=FunctionGroup, 3=Section, 4=Part,
  5=NonStocked, 6=Link); profile applicability via `ComponentConditions.fkProfile`
  (Section level only). Exploded-view graphics are external `GR-#####` CGM refs.
- **imagerepository** (not attached): holds the ~92k images referenced by filename.

## Doc-ref reality check

`D0088`-style ids exist **nowhere** in 2014D (verified incl. VIDA's own token index;
`d00e`/`d039` DTC tokens exist, so the index would have caught it). The doc-ref scheme
is `VCC-nnnnnn-n`; `vida_doc_ref` in our store carries `vccNumber`.

## Extraction recipe (what `src/extract_docs.py` implements)

1. `SELECT id, chronicleId, projectDocumentId, vccNumber, title, fkDocumentType, fkQualifier, XmlContent FROM Document`
2. blob → ZIP → `<chronicleId>_en-US.xml` → lxml
3. normalize to markdown (headings from title/stepgrp/procstep, lists, torque lines,
   xref → target title text); collect `<graphic>` hrefs as image_refs
4. profile tags: `DocumentProfile` → BaseData lookups → distinct model/year/engine
5. section = QualifierGroup.name; tree breadcrumb from TreeItem prepended to text
