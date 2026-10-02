import json

import pytest

from config import ConfigurationError, require_contact_user_agent
from ingestion import clean_dedupe, ofac_sdn


def sdn_row(ent, name, typ, program="IRAN", vess_type="-0-", flag="-0-", remarks="-0-"):
    return {"ent_num": ent, "sdn_name": name, "sdn_type": typ, "program": program, "title": "-0-",
            "call_sign": "-0-", "vess_type": vess_type, "tonnage": "-0-", "grt": "-0-",
            "vess_flag": flag, "vess_owner": "-0-", "remarks": remarks}


def test_ofac_drops_individuals_and_blank_types_at_ingestion():
    rows = [sdn_row("1", "ACME TRADING", "entity"), sdn_row("2", "SOME PERSON", "individual"),
            sdn_row("3", "LEGACY ROW", "-0-"), sdn_row("4", "NO TYPE", "")]
    docs = ofac_sdn.normalize(rows, [], [])
    assert [d.doc_id for d in docs] == ["ofac-sdn-1"]


def test_ofac_placeholders_do_not_leak_into_text():
    row = sdn_row("5", "VESSEL ONE", "vessel", vess_type="Tug", flag="-0-", remarks="Linked To: ACME.")
    alias = [{"ent_num": "5", "alt_num": "1", "alt_type": "aka", "alt_name": "-0-", "alt_remarks": ""}]
    addr = [{"ent_num": "5", "add_num": "1", "address": "-0-", "city_state_prov_postal": "-0-",
             "country": "-0-", "add_remarks": ""}]
    text = ofac_sdn.normalize([row], addr, alias)[0].text
    assert "-0-" not in text
    assert "Vessel type: Tug" in text and "flag:" not in text
    assert "Address(es)" not in text and "Also known as" not in text


def test_ofac_entity_without_vessel_fields_has_no_vessel_line():
    assert "Vessel" not in ofac_sdn.normalize([sdn_row("6", "ACME", "entity")], [], [])[0].text


def make(doc_id, source, text):
    return {"doc_id": doc_id, "source": source, "text": text, "title": doc_id}


def test_boilerplate_stripper_keeps_ofac_field_lines_but_strips_prose_boilerplate():
    ofac = [make(f"o{i}", "OFAC_SDN", f"Name: N{i}\nType: vessel\nProgram(s): IRAN") for i in range(10)]
    sec = [make(f"s{i}", "SEC_EDGAR", f"CONFIDENTIAL HEADER\nunique body {i}") for i in range(10)]
    out = clean_dedupe.strip_boilerplate(ofac + sec)
    assert all("Type: vessel" in d["text"] for d in out if d["source"] == "OFAC_SDN")
    assert all("CONFIDENTIAL HEADER" not in d["text"] for d in out if d["source"] == "SEC_EDGAR")


def test_html_is_stripped_and_exact_duplicates_removed():
    docs = [make("a", "SEC_EDGAR", "<HTML><P class='x'>Hello <b>world</b></P></HTML>"),
            make("b", "SEC_EDGAR", "plain"), make("c", "SEC_EDGAR", "plain ")]
    assert [d["doc_id"] for d in clean_dedupe._remove_exact_dupes_only(docs)] == ["a", "b"]
    assert clean_dedupe.strip_html(docs[:1])[0]["text"] == "Hello world"


def test_near_duplicates_are_tagged_not_dropped():
    base = " ".join(f"word{i}" for i in range(60))
    docs = clean_dedupe.dedupe([make("a", "S", base), make("b", "S", base + " extra"),
                                make("c", "S", "completely different " * 20)])
    assert [d["duplicate_of"] for d in docs] == [None, "a", None]


def test_clean_dedupe_run_end_to_end_on_synthetic_files(tmp_path):
    (tmp_path / "x_normalized.jsonl").write_text("\n".join(
        json.dumps(make(f"d{i}", "OFAC_SDN", f"Name: N{i}\nType: entity")) for i in range(5)))
    stats = clean_dedupe.run(tmp_path)
    assert stats["total_raw"] == 5 and stats["final_corpus_size"] == 5
    assert (tmp_path / "corpus_final.jsonl").exists()


def test_clean_dedupe_without_inputs_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingestion"):
        clean_dedupe.run(tmp_path)


def test_contact_user_agent_required_with_actionable_message(settings):
    with pytest.raises(ConfigurationError, match="CONTACT_USER_AGENT"):
        require_contact_user_agent(settings)
