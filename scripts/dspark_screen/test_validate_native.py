import copy
import unittest

from validate_native import check_state


class NativeEvidenceTests(unittest.TestCase):
    def fixture(self, proposed=None, verified=None, emitted=None):
        proposed = proposed or [31, 32, 33]
        verified = verified or [31, 99]
        emitted = emitted or verified
        binding = {"target_embedding_identity": "0x100", "target_head_identity": "0x200",
                   "embedding_hash_fnv1a64": 100, "head_hash_fnv1a64": 200,
                   "target_embedding_bytes": 777912320, "target_head_bytes": 777912320,
                   "target_embedding_dtype": "f16", "target_head_dtype": "f16",
                   "borrows_embedding": True, "borrows_head": True,
                   "draft_n_batch": 32, "draft_n_ubatch": 32, "draft_n_outputs_max": 7,
                   "draft_n_outputs_max_per_seq": 7, "draft_backend_sampling": True}
        state = [{"event": "binding_begin", **binding},
                 {"event": "inject", "rc": 0, "target_taps": [2, 10, 18, 26, 34],
                  "feature_hash_fnv1a64": 88, "first_position": 0, "last_position": 1, "n_tokens": 2},
                 {"event": "noise", "n_noise_tokens": 7, "first_read_slot": 0,
                  "sample_from_anchor": True, "all_noise_rows_bidirectional": True,
                  "n_proposal_requested": 3, "mask_token_id": 151669, "kv_max_before": 1,
                  "anchor_position": 2, "anchor_token_id": 20, "prefix_token_ids": [10, 11]},
                 {"event": "binding_end", **binding}]
        state[3:3] = [{"event": "mask", "query_position": i, "anchor_position": 2,
                       "visible_noise_positions": list(range(2, 9)), "max_visible_clean_position": 1}
                      for i in range(2, 9)]
        accepted = len(verified) - 1
        rounds = [{"status": "complete", "replay": False, "proposed_token_ids": proposed,
                   "verified_token_ids": verified, "emitted_token_ids": emitted,
                   "n_accepted": accepted, "n_proposed": len(proposed), "n_emitted": len(emitted),
                   "n_accepted_usable_prefix": min(accepted, len(emitted))}]
        return state, rounds

    def test_zero_partial_full_acceptance(self):
        for verified in ([99], [31, 99], [31, 32, 33, 99]):
            state, rounds = self.fixture(verified=verified)
            check_state(state, rounds, 3)

    def test_accepted_eos_preserves_verified_tail(self):
        state, rounds = self.fixture([31, 151645, 33], [31, 151645, 77], [31, 151645])
        _, summary = check_state(state, rounds, 3)
        self.assertEqual(summary["eos_rounds"], 1)

    def test_reject_changed_target_hash(self):
        state, rounds = self.fixture()
        state[-1]["head_hash_fnv1a64"] += 1
        with self.assertRaisesRegex(ValueError, "binding changed"):
            check_state(state, rounds, 3)

    def test_reject_incomplete_lifetime(self):
        state, rounds = self.fixture()
        with self.assertRaisesRegex(ValueError, "binding lifetime"):
            check_state(state[:-1], rounds, 3)

    def test_reject_short_noise_and_slot_one(self):
        for key, value in (("n_noise_tokens", 3), ("first_read_slot", 1)):
            state, rounds = self.fixture()
            state[2][key] = value
            with self.assertRaisesRegex(ValueError, "noise/read-slot"):
                check_state(state, rounds, 3)

    def test_reject_actual_mask_with_hidden_future_row(self):
        state, rounds = self.fixture()
        state[3]["visible_noise_positions"] = [2]
        with self.assertRaisesRegex(ValueError, "actual attention mask"):
            check_state(state, rounds, 3)

    def test_reject_old_four_output_capacity(self):
        state, rounds = self.fixture()
        for binding in (state[0], state[-1]):
            binding["draft_n_outputs_max_per_seq"] = 4
        with self.assertRaisesRegex(ValueError, "capacity cannot cover"):
            check_state(state, rounds, 3)

    def test_reject_zero_drafts_and_permanently_truncated_drafts(self):
        for proposed, verified in (([], [99]), ([31], [31, 99])):
            state, rounds = self.fixture()
            rounds[0].update(proposed_token_ids=proposed, verified_token_ids=verified,
                             n_proposed=len(proposed), n_accepted=len(verified)-1,
                             n_accepted_usable_prefix=min(len(verified)-1, len(rounds[0]["emitted_token_ids"])))
            with self.assertRaisesRegex(ValueError, "configured maximum"):
                check_state(state, rounds, 3)

    def test_legitimate_no_noise_output_cap_boundary_is_separate(self):
        state, rounds = self.fixture()
        rounds.append({"status": "complete", "replay": False, "n_draft_max": 0,
                       "n_proposed": 0, "n_accepted": 0, "proposed_token_ids": [],
                       "n_emitted": 1, "emitted_token_ids": [151645]})
        _, summary = check_state(state, rounds, 3)
        self.assertEqual(summary["rounds"], 1)
        self.assertEqual(summary["no_noise_output_cap_boundaries"], 1)

    def test_reject_stale_noise_cache(self):
        state, rounds = self.fixture()
        state[2]["kv_max_before"] = 8
        with self.assertRaisesRegex(ValueError, "stale noise"):
            check_state(state, rounds, 3)

    def test_reject_nonprefix_verifier(self):
        state, rounds = self.fixture()
        rounds[0]["verified_token_ids"][0] = 88
        with self.assertRaisesRegex(ValueError, "matched-prefix"):
            check_state(state, rounds, 3)

    def test_reject_emission_after_eos(self):
        state, rounds = self.fixture([31, 151645, 33], [31, 151645, 77], [31, 151645, 77])
        with self.assertRaisesRegex(ValueError, "after EOS"):
            check_state(state, rounds, 3)

    def test_require_reinjection_before_next_anchor(self):
        state, rounds = self.fixture()
        next_noise = copy.deepcopy(state[2])
        next_noise.update(anchor_position=4, prefix_token_ids=[10, 11, 20, 31], kv_max_before=3,
                          anchor_token_id=99)
        state.insert(-1, next_noise)
        state[-1:-1] = [{"event": "mask", "query_position": i, "anchor_position": 4,
                          "visible_noise_positions": list(range(4, 11)), "max_visible_clean_position": 3}
                         for i in range(4, 11)]
        rounds.append(copy.deepcopy(rounds[0]))
        with self.assertRaisesRegex(ValueError, "not reinjected"):
            check_state(state, rounds, 3)
        state.insert(-9, {"event": "inject", "rc": 0, "target_taps": [2, 10, 18, 26, 34],
                          "feature_hash_fnv1a64": 89, "first_position": 2, "last_position": 5, "n_tokens": 4})
        check_state(state, rounds, 3)


if __name__ == "__main__":
    unittest.main()
