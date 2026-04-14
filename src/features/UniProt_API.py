"""Utilities for fetching and parsing UniProt records through the REST API.

This module is intentionally UniProt-only. It fetches one UniProtKB record,
parses the REST JSON payload, and exposes biologically useful fields through a
small wrapper class. The code is written in a teaching-friendly style:

    fetch -> inspect JSON -> normalize -> expose through getters
"""

import re
import time
import requests

from ..utils.logger import LOGGER


REST_UNIPROT = "https://rest.uniprot.org/uniprotkb/{}"

comma_splitter = re.compile(r"\s*,\s*").split

class UniprotRecord(object):
    """Wrap one UniProt REST record and expose parsed biological annotations."""

    def __init__(self, data):
        self._rawdata = data
        self._pdbdata = []
        self._parse()

    def __repr__(self):
        return "<UniprotRecord: %s>" % self.getTitle()

    def __str__(self):
        return self.getTitle()

    def getAPIURL(self):
        """Return the REST API URL for the current UniProt accession."""
        return REST_UNIPROT.format(self.getAccession())

    def getData(self):
        """Return the raw UniProt REST JSON dictionary."""
        return self._rawdata

    def getPDBs(self):
        """Return UniProt PDB cross-references parsed from the REST payload."""
        return self._pdbdata

    def getAccession(self):
        """Return the primary UniProt accession."""
        return self._rawdata.get("primaryAccession")

    def getSecondaryAccessions(self):
        """Return the list of secondary UniProt accessions."""
        return self._rawdata.get("secondaryAccessions", [])

    def getName(self):
        """Return the UniProtKB entry name, such as HLAA_HUMAN."""
        return self._rawdata.get("uniProtkbId")

    def getProtein(self):
        """Parse recommended and alternative protein names."""
        desc = self._rawdata.get("proteinDescription", {})
        recommended = (
            desc.get("recommendedName", {})
            .get("fullName", {})
            .get("value")
        )
        alternatives = desc.get("alternativeNames", [])
        alt_full = None
        alt_short = None
        if alternatives:
            alt_full = alternatives[0].get("fullName", {}).get("value")
            shorts = alternatives[0].get("shortNames", [])
            if shorts:
                alt_short = shorts[0].get("value")
        return {
            "recommend_name": recommended,
            "alter_fullname": alt_full,
            "alter_shortname": alt_short,
        }

    def getGene(self):
        """Parse the primary gene name from the genes block."""
        genes = self._rawdata.get("genes", [])
        if not genes:
            return None
        return genes[0].get("geneName", {}).get("value")

    def getGenes(self):
        """Return all parsed gene names and synonyms."""
        genes_out = []
        for gene in self._rawdata.get("genes", []):
            genes_out.append(
                {
                    "gene_name": (gene.get("geneName") or {}).get("value"),
                    "synonyms": [item.get("value") for item in gene.get("synonyms", []) if item.get("value")],
                }
            )
        return genes_out

    def getOrganism(self):
        """Parse organism names, taxonomy id, and lineage."""
        organism = self._rawdata.get("organism", {})
        return {
            "scientific_name": organism.get("scientificName"),
            "common_name": organism.get("commonName"),
            "taxonomy_id": organism.get("taxonId"),
            "lineage": organism.get("lineage", []),
        }

    def getSequence(self):
        """Return the amino-acid sequence string."""
        return (self._rawdata.get("sequence") or {}).get("value")

    def getCellLocation(self):
        """Return parsed subcellular and membrane-related location annotations."""
        return self._cell_location

    def getDNAbinding(self):
        """Return parsed DNA-binding region features."""
        return self._dna_binding

    def getZincFinger(self):
        """Return parsed zinc-finger features."""
        return self._zinc_finger

    def getActiveSite(self):
        """Return parsed active-site features."""
        return self._active_site

    def getBindingSite(self):
        """Return parsed binding-site features and ligands."""
        return self._binding_site

    def getSite(self):
        """Return parsed generic site features."""
        return self._site

    def getAlphaFold(self):
        """Return the AlphaFoldDB accession cross-reference, if present."""
        hits = self._xrefs("AlphaFoldDB")
        return hits[0].get("id") if hits else None

    def getCofactor(self):
        """Return parsed cofactor comment annotations."""
        return self._cofactors

    def getPTMProcessing(self):
        """Return parsed PTM and processing annotations."""
        return self._ptm_processing

    def getInteractions(self):
        """Return parsed protein-protein interaction partners."""
        return self._interactions

    def getFamilyDomains(self):
        """Return parsed domain, region, motif, and repeat features."""
        return self._family_domains

    def getComplexPortals(self):
        """Return parsed ComplexPortal cross-references."""
        return self._complex_portals

    def getSubunit(self, kind="all"):
        """Return parsed subunit comment text, optionally microbial-only."""
        kind = (kind or "all").lower()
        if kind not in ("all", "microbial", "non_microbial"):
            kind = "all"
        return self._subunit_comments.get(kind, [])

    def getTitle(self):
        """Return a readable title combining accession and entry name."""
        uid = self.getAccession()
        name = self.getName()
        return f"{uid} ({name})"

    def getEntry(self, item):
        """Backward-compatible accessor for a few common UniProt fields."""
        if item == "accession":
            return self.getAccession()
        if item == "name":
            return self.getName()
        if item == "sequence":
            return self.getSequence()
        return self._rawdata.get(item)

    def getKeywords(self):
        """Return parsed keyword values from the UniProt keyword list."""
        return self._keywords

    def getReferences(self):
        """Return literature references reported by UniProt."""
        return self._references

    def getFeatures(self):
        """Return the raw UniProt feature list for complete inspection."""
        return self._features

    def getComments(self):
        """Return the raw UniProt comment list for complete inspection."""
        return self._comments

    def getCrossReferences(self):
        """Return cross-references grouped by source database."""
        return self._cross_references

    def _text_values(self, block):
        """Extract plain text values from a UniProt comment block."""
        texts = []
        for item in (block or {}).get("texts", []):
            value = item.get("value")
            if value:
                texts.append(value.strip())
        return texts

    def _start_end(self, location):
        """Extract start and end coordinates from a UniProt location block."""
        location = location or {}
        start = (location.get("start") or {}).get("value")
        end = (location.get("end") or {}).get("value")
        return start, end

    def _position_or_range(self, location):
        """Convert a location block into a single position or begin-end range."""
        start, end = self._start_end(location)
        if start is None and end is None:
            return None
        if start == end:
            return start
        return f"{start}-{end}"

    def _xref_properties(self, xref):
        """Convert cross-reference property list into a simple dictionary."""
        out = {}
        for item in (xref or {}).get("properties", []):
            key = item.get("key")
            value = item.get("value")
            if key:
                out[key] = value
        return out

    def _xrefs(self, database):
        """Return all cross-references for one database, such as PDB or AlphaFoldDB."""
        return [
            x
            for x in self._rawdata.get("uniProtKBCrossReferences", [])
            if isinstance(x, dict) and x.get("database") == database
        ]

    def _parseDNAbinding(self):
        """Parse DNA-binding region features from the features block."""
        dna_binding = []
        for feature in self._rawdata.get("features", []):
            if (feature.get("type") or "").strip().lower() != "dna binding":
                continue
            start, end = self._start_end(feature.get("location"))
            dna_binding.append(
                {
                    "description": feature.get("description"),
                    "begin": start,
                    "end": end,
                }
            )
        self._dna_binding = dna_binding

    def _parseZincfinger(self):
        """Parse zinc-finger features from the features block."""
        zinc_finger = []
        for feature in self._rawdata.get("features", []):
            if (feature.get("type") or "").strip().lower() != "zinc finger":
                continue
            start, end = self._start_end(feature.get("location"))
            zinc_finger.append(
                {
                    "description": feature.get("description"),
                    "begin": start,
                    "end": end,
                }
            )
        self._zinc_finger = zinc_finger

    def _parseActiveSite(self):
        """Parse active-site features from the features block."""
        active_site = []
        for feature in self._rawdata.get("features", []):
            if (feature.get("type") or "").strip().lower() != "active site":
                continue
            active_site.append(
                {
                    "description": feature.get("description"),
                    "position": self._position_or_range(feature.get("location")),
                }
            )
        self._active_site = active_site

    def _parseBindingSite(self):
        """Parse binding-site features and associated ligand annotations."""
        binding_site = []
        for feature in self._rawdata.get("features", []):
            if (feature.get("type") or "").strip().lower() != "binding site":
                continue
            chebi = None
            for ref in feature.get("featureCrossReferences", []):
                if ref.get("database") == "ChEBI":
                    chebi = ref.get("id")
                    break
            ligand = feature.get("ligand") or {}
            binding_site.append(
                {
                    "position": self._position_or_range(feature.get("location")),
                    "description": feature.get("description"),
                    "name": ligand.get("name"),
                    "chebi": chebi or ligand.get("id"),
                }
            )
        self._binding_site = binding_site

    def _parseSite(self):
        """Parse generic site features from the features block."""
        site = []
        for feature in self._rawdata.get("features", []):
            if (feature.get("type") or "").strip().lower() != "site":
                continue
            site.append(
                {
                    "position": self._position_or_range(feature.get("location")),
                    "description": feature.get("description"),
                }
            )
        self._site = site

    def _parseCofactor(self):
        """Parse cofactors from UniProt COFACTOR comments."""
        cofactors = []
        for comment in self._rawdata.get("comments", []):
            if (comment.get("commentType") or "").strip().upper() != "COFACTOR":
                continue
            for item in comment.get("cofactors", []):
                chebi = None
                xref = item.get("cofactorCrossReference") or {}
                if xref.get("database") == "ChEBI":
                    chebi = xref.get("id")
                cofactors.append(
                    {
                        "name": item.get("name"),
                        "chebi": chebi,
                    }
                )
        self._cofactors = cofactors

    def _parseCellLocation(self):
        """Parse membrane regions and subcellular-location comments."""
        cell_location = []
        for feature in self._rawdata.get("features", []):
            ftype = (feature.get("type") or "").strip().lower()
            if ftype not in (
                "topological domain",
                "transmembrane",
                "intramembrane",
            ):
                continue
            start, end = self._start_end(feature.get("location"))
            cell_location.append(
                {
                    "type": feature.get("type"),
                    "description": feature.get("description"),
                    "begin": start,
                    "end": end,
                }
            )
        for comment in self._rawdata.get("comments", []):
            if (comment.get("commentType") or "").strip().upper() != "SUBCELLULAR LOCATION":
                continue
            for item in comment.get("subcellularLocations", []):
                location = (item.get("location") or {}).get("value")
                topology = (item.get("topology") or {}).get("value")
                if location or topology:
                    cell_location.append(
                        {
                            "type": "Subcellular location",
                            "description": location,
                            "begin": None,
                            "end": None,
                            "topology": topology,
                        }
                    )
        self._cell_location = cell_location

    def _parsePTMProcessing(self):
        """Parse PTM-related features and PTM comment text."""
        ptms = []
        feature_types = {
            "signal",
            "transit peptide",
            "chain",
            "propeptide",
            "peptide",
            "disulfide bond",
            "glycosylation",
            "modified residue",
            "lipidation",
        }
        for feature in self._rawdata.get("features", []):
            if (feature.get("type") or "").strip().lower() not in feature_types:
                continue
            start, end = self._start_end(feature.get("location"))
            if start == end:
                ptms.append(
                    {
                        "type": feature.get("type"),
                        "position": start,
                        "description": feature.get("description"),
                    }
                )
            else:
                ptms.append(
                    {
                        "type": feature.get("type"),
                        "begin": start,
                        "end": end,
                        "description": feature.get("description"),
                    }
                )
        for comment in self._rawdata.get("comments", []):
            if (comment.get("commentType") or "").strip().upper() != "PTM":
                continue
            for text in self._text_values(comment):
                ptms.append(
                    {
                        "type": "PTM comment",
                        "description": text,
                    }
                )
        self._ptm_processing = ptms

    def _parseComplexPortals(self):
        """Parse ComplexPortal cross-references from the xref list."""
        portals = []
        for xref in self._xrefs("ComplexPortal"):
            props = self._xref_properties(xref)
            portals.append(
                {
                    "id": xref.get("id"),
                    "name": props.get("EntryName"),
                }
            )
        self._complex_portals = portals

    def _parseSubunit(self):
        """Parse SUBUNIT comments and split microbial from non-microbial text."""
        subunit_all = []
        for comment in self._rawdata.get("comments", []):
            if (comment.get("commentType") or "").strip().upper() != "SUBUNIT":
                continue
            subunit_all.extend(self._text_values(comment))
        microbial = [t for t in subunit_all if "(Microbial infection)" in t]
        non_micro = [t for t in subunit_all if "(Microbial infection)" not in t]
        self._subunit_comments = {
            "all": subunit_all,
            "microbial": microbial,
            "non_microbial": non_micro,
        }

    def _parseFamilyDomains(self):
        """Parse domains, regions, motifs, coiled coils, and repeats."""
        family_domains = []
        feature_types = {
            "domain",
            "region",
            "coiled coil",
            "short sequence motif",
            "repeat",
        }
        for feature in self._rawdata.get("features", []):
            if (feature.get("type") or "").strip().lower() not in feature_types:
                continue
            start, end = self._start_end(feature.get("location"))
            family_domains.append(
                {
                    "type": feature.get("type"),
                    "description": feature.get("description"),
                    "begin": start,
                    "end": end,
                }
            )
        self._family_domains = family_domains

    def _parseInteractions(self):
        """Parse interaction partners from INTERACTION comments."""
        interactions = []
        self_acc = self.getAccession()
        for comment in self._rawdata.get("comments", []):
            if (comment.get("commentType") or "").strip().upper() != "INTERACTION":
                continue
            for item in comment.get("interactions", []):
                one = item.get("interactantOne") or {}
                two = item.get("interactantTwo") or {}
                if one.get("uniProtKBAccession") == self_acc:
                    partner = two
                else:
                    partner = one
                partner_acc = partner.get("uniProtKBAccession")
                if not partner_acc or partner_acc == self_acc:
                    continue
                interactions.append(
                    {
                        "id": partner_acc,
                        "label": partner.get("geneName"),
                    }
                )
        self._interactions = interactions

    def _parsePDBfromUniProt(self):
        """Parse PDB cross-references reported directly by UniProt."""
        pdbdata = {}
        for xref in self._xrefs("PDB"):
            pdbid = xref.get("id")
            props = self._xref_properties(xref)
            method = props.get("Method")
            resolution = props.get("Resolution", "1.00 A")
            try:
                resolution = float(str(resolution).split(" ")[0])
            except Exception:
                resolution = None
            pdbchains = props.get("Chains", "")
            chains = []
            resrange = None
            try:
                for chain in comma_splitter(pdbchains):
                    chids, resrange = chain.split("=")
                    chids = [chid.strip() for chid in chids.split("/")]
                    chains.extend(chids)
            except Exception as exc:
                LOGGER.warn(str(exc))
                LOGGER.warn("Suspected no chain information")
            pdbdata[pdbid] = {
                "method": method,
                "resolution": resolution,
                "chains": chains,
                "resrange": resrange,
            }
        self._pdbdata = pdbdata

    def _parseKeywords(self):
        """Parse UniProt keywords as a simple list of strings."""
        self._keywords = [
            item.get("name")
            for item in self._rawdata.get("keywords", [])
            if isinstance(item, dict) and item.get("name")
        ]

    def _parseReferences(self):
        """Parse literature references with citation metadata."""
        refs = []
        for item in self._rawdata.get("references", []):
            citation = item.get("citation") or {}
            refs.append(
                {
                    "id": item.get("citationId"),
                    "type": citation.get("citationType"),
                    "title": citation.get("title"),
                    "journal": citation.get("journal"),
                    "volume": citation.get("volume"),
                    "first_page": citation.get("firstPage"),
                    "last_page": citation.get("lastPage"),
                    "year": citation.get("publicationDate"),
                    "authors": citation.get("authors", []),
                }
            )
        self._references = refs

    def _parseFeatures(self):
        """Keep the full UniProt features list for downstream inspection."""
        self._features = list(self._rawdata.get("features", []))

    def _parseComments(self):
        """Keep the full UniProt comments list for downstream inspection."""
        self._comments = list(self._rawdata.get("comments", []))

    def _parseCrossReferences(self):
        """Group cross-references by source database name."""
        grouped = {}
        for xref in self._rawdata.get("uniProtKBCrossReferences", []):
            if not isinstance(xref, dict):
                continue
            db = xref.get("database")
            if not db:
                continue
            grouped.setdefault(db, []).append(xref)
        self._cross_references = grouped

    def _parse(self):
        """Parse the main UniProt fields into convenient cached attributes."""
        acc = self.getAccession()
        LOGGER.info(f"Parse UniProt information of {acc}...")
        LOGGER.timeit("_parse")
        self._parseComments()
        self._parseFeatures()
        self._parseCrossReferences()
        self._parseKeywords()
        self._parseReferences()
        self._parseActiveSite()
        self._parseBindingSite()
        self._parseSite()
        self._parseCofactor()
        self._parseDNAbinding()
        self._parseZincfinger()
        self._parseCellLocation()
        self._parsePTMProcessing()
        self._parseFamilyDomains()
        self._parseInteractions()
        self._parseComplexPortals()
        self._parsePDBfromUniProt()
        self._parseSubunit()
        LOGGER.report("Parsing in %.1fs.", "_parse")

def queryUniprot(accession, timeout=20):
    """Fetch one UniProtKB REST record as a raw JSON dictionary."""
    if not isinstance(accession, str):
        raise TypeError("accession should be a string")

    accession = accession.strip()
    if not accession:
        raise ValueError("accession should not be empty")

    url = REST_UNIPROT.format(accession)
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"No UniProt record found for accession {accession}") from exc
    return response.json()

def searchUniprot(accession, timeout=20, n_attempts=3, dt=1):
    """Fetch one UniProt record and wrap it as a :class:`UniprotRecord`."""
    last_error = None
    for attempt in range(n_attempts):
        try:
            data = queryUniprot(accession, timeout=timeout)
            return UniprotRecord(data)
        except Exception as exc:
            last_error = exc
            LOGGER.info(f"Attempt {attempt} to contact UniProt failed")
            if attempt < n_attempts - 1:
                time.sleep((attempt + 1) * dt)
    raise last_error
