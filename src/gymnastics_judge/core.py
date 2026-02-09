import os
from typing import Protocol, Any, Dict, List, Optional
from google import genai
from google.genai import types
from .config import Config

class Tool(Protocol):
    """Protocol that all analysis tools must implement"""
    name: str
    description: str
    
    async def analyze(self, input_path: str) -> Dict[str, Any]:
        """Perform analysis on the input and return structured data"""
        ...

class JudgeAgent:
    def __init__(self):
        Config.validate()
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        # Try different model name formats - common ones are:
        # "gemini-1.5-flash-latest", "gemini-2.0-flash-exp", "gemini-1.5-pro"
        self.model_id = Config.GEMINI_MODEL or "gemini-1.5-flash-latest"

    def _model_names_to_try(self) -> list[str]:
        # Try multiple model ids. Some older aliases (e.g. gemini-1.5-*-latest) may 404 for
        # certain API versions/keys. Prefer stable, currently documented ids.
        #
        # Docs reference examples like:
        # - gemini-2.5-flash
        # - gemini-2.0-flash-001
        # See: https://googleapis.github.io/python-genai/
        candidates = [
            self.model_id,
            # Common stable IDs
            "gemini-2.5-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash-001",
            # Back-compat aliases (may 404; keep as last resort)
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-2.0-flash-exp",
        ]
        # De-duplicate while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for m in candidates:
            if not m or m in seen:
                continue
            seen.add(m)
            out.append(m)
        return out

    async def review_hand_support(
        self,
        video_path: str,
        hold_window_1s: Dict[str, Any],
        *,
        max_sampled_frames: int = 24,
    ) -> Dict[str, Any]:
        """
        Use Gemini vision to decide if a hand touches the ground DURING the hold window.

        We intentionally do NOT rely on MediaPipe hand landmarks for this (videos can be low-res / occluded).
        Instead, we sample frames from the hold window (by frame indices) and ask the LLM to judge contact.
        """
        import os
        import json

        # IMPORTANT: hands may touch the floor during entry/exit, but the rule only cares
        # about contact DURING the hold. We therefore review a ~1s window from *inside*
        # the CV-defined hold window, biased toward the middle of the hold segment to
        # avoid entry/exit frames.
        start_frame = hold_window_1s.get("window_start_frame")
        end_frame = hold_window_1s.get("window_end_frame")
        peak_frame = hold_window_1s.get("peak_frame_within_hold")

        if not (isinstance(start_frame, int) and isinstance(end_frame, int) and end_frame >= start_frame):
            return {
                "hand_support_detected": None,
                "confidence": 0.0,
                "notes": "No valid hold window frames available for LLM video review.",
                "evidence_frame_numbers": [],
                "used_model": None,
                "used_frames": {"start_frame": start_frame, "end_frame": end_frame, "sampled": 0},
            }

        if not os.path.exists(video_path):
            return {
                "hand_support_detected": None,
                "confidence": 0.0,
                "notes": f"Video not found for LLM review: {video_path}",
                "evidence_frame_numbers": [],
                "used_model": None,
                "used_frames": {"start_frame": start_frame, "end_frame": end_frame, "sampled": 0},
            }

        # Sample frames from the hold window to keep requests small/fast.
        try:
            import cv2  # local import so core can still import without cv2 in non-CV contexts
        except Exception as e:
            return {
                "hand_support_detected": None,
                "confidence": 0.0,
                "notes": f"OpenCV (cv2) not available for extracting frames: {e}",
                "evidence_frame_numbers": [],
                "used_model": None,
                "used_frames": {"start_frame": start_frame, "end_frame": end_frame, "sampled": 0},
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "hand_support_detected": None,
                "confidence": 0.0,
                "notes": f"Could not open video for LLM review: {video_path}",
                "evidence_frame_numbers": [],
                "used_model": None,
                "used_frames": {"start_frame": start_frame, "end_frame": end_frame, "sampled": 0},
            }

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        center_frame = int((start_frame + end_frame) // 2)
        if not isinstance(peak_frame, int):
            peak_frame = center_frame

        # Use the *center* of the hold for hand review (more robust to "hands touch at beginning/end").
        review_center = center_frame

        # Review window: wide enough to capture a full 1s hands-off interval even if
        # the hand lifts during the window (entry/exit).
        # Use ~3 seconds total (±1.5s) centered in-hold.
        review_half_window_seconds = 1.5
        half_window_frames = int(round(fps * review_half_window_seconds)) if fps > 0 else 45
        review_start = max(int(start_frame), int(review_center) - half_window_frames)
        review_end = min(int(end_frame), int(review_center) + half_window_frames)
        if review_end < review_start:
            review_start, review_end = int(start_frame), int(end_frame)

        # Use fewer, clearer frames with helpful zooms rather than many tiny frames.
        max_sampled_frames = min(int(max_sampled_frames), 18)
        total = (review_end - review_start + 1)
        step = max(1, int(total / max(1, max_sampled_frames)))
        sampled_parts: list[types.Part] = []
        sampled_frame_numbers: list[int] = []

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, review_start - 1))
            current_frame_num = review_start
            while current_frame_num <= review_end:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only keep every Nth frame to cap tokens.
                if ((current_frame_num - review_start) % step) == 0:
                    # Downscale for token/size efficiency (keep more detail than before)
                    h, w = frame.shape[:2]
                    target_w = 960
                    if w > target_w:
                        scale = target_w / float(w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                    # Prepare a zoomed crop of the lower half (hands + floor usually live here).
                    h2, w2 = frame.shape[:2]
                    crop = frame[int(h2 * 0.45) : h2, 0:w2].copy()

                    # Overlay frame number to help the model reference evidence consistently
                    cv2.putText(frame, f"FULL frame {current_frame_num}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(crop, f"CROP frame {current_frame_num}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    ok1, buf1 = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    ok2, buf2 = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok1 and ok2:
                        # Provide both full context and zoom for each sampled timepoint.
                        sampled_parts.append(types.Part.from_bytes(data=buf1.tobytes(), mime_type="image/jpeg"))
                        sampled_parts.append(types.Part.from_bytes(data=buf2.tobytes(), mime_type="image/jpeg"))
                        sampled_frame_numbers.append(int(current_frame_num))

                current_frame_num += 1
        finally:
            cap.release()

        if not sampled_parts:
            return {
                "hand_support_detected": None,
                "confidence": 0.0,
                "notes": "Failed to extract any frames for LLM review.",
                "evidence_frame_numbers": [],
                "used_model": None,
                "used_frames": {"start_frame": start_frame, "end_frame": end_frame, "sampled": 0},
            }

        prompt = (
            "You are an expert judge reviewing HAND SUPPORT during a Penche HOLD.\n"
            "You will receive frames from a ~3 second REVIEW WINDOW taken from INSIDE the hold.\n"
            "This window is centered in the MIDDLE of the hold segment to avoid entry/exit.\n"
            "Do NOT judge hand contact that happens outside this review window.\n\n"
            "For each sampled moment you will receive TWO images:\n"
            "- a FULL frame\n"
            "- a ZOOMED CROP of the lower half (hands + floor)\n\n"
            "Images are in repeating pairs: FULL then CROP, for the same frame number.\n"
            "Use the CROP to decide contact.\n\n"
            "For EACH sampled frame number, classify it as one of:\n"
            "- TOUCH: clear hand/fingers/forearm contact with the floor (no visible gap)\n"
            "- NO_TOUCH: clear air gap / clearly not touching\n"
            "- UNCLEAR: cannot tell due to blur/occlusion/shadows\n\n"
            "Be conservative:\n"
            "- Do NOT assume touch from proximity.\n"
            "- If UNCLEAR, prefer UNCLEAR (not TOUCH).\n\n"
            f"Hold window frames: {start_frame}..{end_frame}\n"
            f"Peak frame (within hold): {peak_frame}\n"
            f"Hold center frame: {center_frame}\n"
            f"Review window frames: {review_start}..{review_end}\n"
            f"Sampled frame numbers (chronological): {sampled_frame_numbers}\n"
            "Return ONLY JSON matching the schema.\n"
        )

        schema = {
            "type": "OBJECT",
            "required": ["frame_statuses", "confidence", "notes"],
            "properties": {
                "frame_statuses": {
                    "type": "ARRAY",
                    "items": {"type": "STRING", "enum": ["TOUCH", "NO_TOUCH", "UNCLEAR"]},
                },
                "confidence": {"type": "NUMBER", "description": "0.0 to 1.0"},
                "notes": {"type": "STRING"},
            },
        }

        # Prefer a stronger vision model for subtle contact judgment; fall back to others if unavailable.
        last_error: Exception | None = None
        model_try_order = ["gemini-2.5-pro", *self._model_names_to_try()]
        seen_models: set[str] = set()
        for model_name in model_try_order:
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=[types.Part.from_text(text=prompt), *sampled_parts],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )

                parsed = getattr(response, "parsed", None)
                if parsed is None:
                    parsed = json.loads(response.text or "{}")

                statuses = list(parsed.get("frame_statuses") or [])
                confidence = float(parsed.get("confidence") or 0.0)
                notes = str(parsed.get("notes") or "")

                # Normalize length to sampled_frame_numbers
                if len(statuses) < len(sampled_frame_numbers):
                    statuses.extend(["UNCLEAR"] * (len(sampled_frame_numbers) - len(statuses)))
                if len(statuses) > len(sampled_frame_numbers):
                    statuses = statuses[: len(sampled_frame_numbers)]

                # Compute longest contiguous "safe" run (NO_TOUCH or UNCLEAR).
                # This handles the case where hands touch during entry/exit but not during the actual hold.
                fps_for_est = fps if fps > 0 else 30.0
                best_run = (0, 0)  # indices in sampled_frame_numbers list: [s, e] inclusive
                cur_s = None
                for i, st in enumerate(statuses):
                    safe = st in ("NO_TOUCH", "UNCLEAR")
                    if safe and cur_s is None:
                        cur_s = i
                    if (not safe) and cur_s is not None:
                        if (i - 1) - cur_s > best_run[1] - best_run[0]:
                            best_run = (cur_s, i - 1)
                        cur_s = None
                if cur_s is not None:
                    if (len(statuses) - 1) - cur_s > best_run[1] - best_run[0]:
                        best_run = (cur_s, len(statuses) - 1)

                run_start_frame = int(sampled_frame_numbers[best_run[0]])
                run_end_frame = int(sampled_frame_numbers[best_run[1]])
                run_seconds_est = float(max(0.0, (run_end_frame - run_start_frame) / fps_for_est))

                # If we can find >=1.0s of safe frames, we treat balance as maintained during the hold.
                balance_ok = run_seconds_est >= 1.0

                # Hand support during hold is only considered true if we cannot find a >=1s safe run.
                hand_support_during_hold = None
                if balance_ok:
                    hand_support_during_hold = False
                else:
                    hand_support_during_hold = ("TOUCH" in statuses)

                evidence = [int(f) for f, st in zip(sampled_frame_numbers, statuses) if st == "TOUCH"]

                return {
                    "hand_support_detected": bool("TOUCH" in statuses),
                    "confidence": confidence,
                    "notes": notes,
                    "evidence_frame_numbers": evidence,
                    "frame_statuses": statuses,
                    "balance_maintained_during_hold_estimate": balance_ok,
                    "hands_off_longest_run": {
                        "start_frame": run_start_frame,
                        "end_frame": run_end_frame,
                        "duration_seconds_estimate": run_seconds_est,
                    },
                    "hand_support_detected_during_hold_estimate": hand_support_during_hold,
                    "used_model": model_name,
                    "used_frames": {
                        "hold_start_frame": int(start_frame),
                        "hold_end_frame": int(end_frame),
                        "peak_frame": int(peak_frame),
                        "hold_center_frame": int(center_frame),
                        "review_start_frame": int(review_start),
                        "review_end_frame": int(review_end),
                        "sampled": int(len(sampled_frame_numbers)),
                    },
                }
            except Exception as e:
                last_error = e
                if "404" not in str(e) and "not found" not in str(e).lower():
                    raise
                continue

        return {
            "hand_support_detected": None,
            "confidence": 0.0,
            "notes": f"Could not find a valid Gemini model for vision review. Last error: {last_error}",
            "evidence_frame_numbers": [],
            "used_model": None,
            "used_frames": {"start_frame": start_frame, "end_frame": end_frame, "sampled": int(len(sampled_frame_numbers))},
        }

    async def review_releve(
        self,
        video_path: str,
        hold_window_1s: Dict[str, Any],
        *,
        max_sampled_frames: int = 18,
    ) -> Dict[str, Any]:
        """
        Use Gemini vision to judge RELEVÉ (tip-toe / heel lifted) during the hold.

        MediaPipe-based foot/heel detection is unreliable for low-res/occluded videos,
        so we use a Gemini vision review similar to hand-support.
        """
        import os
        import json

        start_frame = hold_window_1s.get("window_start_frame")
        end_frame = hold_window_1s.get("window_end_frame")

        if not (isinstance(start_frame, int) and isinstance(end_frame, int) and end_frame >= start_frame):
            return {
                "releve_maintained_majority_estimate": None,
                "confidence": 0.0,
                "notes": "No valid hold window frames available for relevé review.",
                "flat_evidence_frame_numbers": [],
                "frame_statuses": [],
                "used_model": None,
                "used_frames": {"hold_start_frame": start_frame, "hold_end_frame": end_frame, "sampled": 0},
            }

        if not os.path.exists(video_path):
            return {
                "releve_maintained_majority_estimate": None,
                "confidence": 0.0,
                "notes": f"Video not found for relevé review: {video_path}",
                "flat_evidence_frame_numbers": [],
                "frame_statuses": [],
                "used_model": None,
                "used_frames": {"hold_start_frame": start_frame, "hold_end_frame": end_frame, "sampled": 0},
            }

        try:
            import cv2
        except Exception as e:
            return {
                "releve_maintained_majority_estimate": None,
                "confidence": 0.0,
                "notes": f"OpenCV (cv2) not available for extracting frames: {e}",
                "flat_evidence_frame_numbers": [],
                "frame_statuses": [],
                "used_model": None,
                "used_frames": {"hold_start_frame": start_frame, "hold_end_frame": end_frame, "sampled": 0},
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "releve_maintained_majority_estimate": None,
                "confidence": 0.0,
                "notes": f"Could not open video for relevé review: {video_path}",
                "flat_evidence_frame_numbers": [],
                "frame_statuses": [],
                "used_model": None,
                "used_frames": {"hold_start_frame": start_frame, "hold_end_frame": end_frame, "sampled": 0},
            }

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # Prefer evaluating within the hands-off sub-hold if available (best match to the rule timing).
        run = hold_window_1s.get("hands_off_longest_run")
        review_start = int(start_frame)
        review_end = int(end_frame)
        if isinstance(run, dict) and isinstance(run.get("start_frame"), int) and isinstance(run.get("end_frame"), int):
            rs, re = int(run["start_frame"]), int(run["end_frame"])
            if re >= rs:
                review_start = max(int(start_frame), rs)
                review_end = min(int(end_frame), re)

        # If we still don't have ~1s span, widen to ~3s centered in-hold.
        span_frames = max(0, review_end - review_start)
        if fps > 0 and (span_frames / fps) < 1.0:
            center = int((int(start_frame) + int(end_frame)) // 2)
            half_window_frames = int(round(fps * 1.5))
            review_start = max(int(start_frame), center - half_window_frames)
            review_end = min(int(end_frame), center + half_window_frames)

        max_sampled_frames = min(int(max_sampled_frames), 18)
        total = (review_end - review_start + 1)
        step = max(1, int(total / max(1, max_sampled_frames)))

        sampled_parts: list[types.Part] = []
        sampled_frame_numbers: list[int] = []

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, review_start - 1))
            current_frame_num = review_start
            while current_frame_num <= review_end:
                ret, frame = cap.read()
                if not ret:
                    break

                if ((current_frame_num - review_start) % step) == 0:
                    # Keep more detail for feet/heels.
                    h, w = frame.shape[:2]
                    target_w = 960
                    if w > target_w:
                        scale = target_w / float(w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                    h2, w2 = frame.shape[:2]
                    bottom = frame[int(h2 * 0.55) : h2, 0:w2].copy()
                    left_half = bottom[:, 0 : (w2 // 2)].copy()
                    right_half = bottom[:, (w2 // 2) : w2].copy()

                    cv2.putText(frame, f"FULL frame {current_frame_num}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(left_half, f"FEET-L frame {current_frame_num}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(right_half, f"FEET-R frame {current_frame_num}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    ok1, buf1 = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    ok2, buf2 = cv2.imencode(".jpg", left_half, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    ok3, buf3 = cv2.imencode(".jpg", right_half, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok1 and ok2 and ok3:
                        sampled_parts.append(types.Part.from_bytes(data=buf1.tobytes(), mime_type="image/jpeg"))
                        sampled_parts.append(types.Part.from_bytes(data=buf2.tobytes(), mime_type="image/jpeg"))
                        sampled_parts.append(types.Part.from_bytes(data=buf3.tobytes(), mime_type="image/jpeg"))
                        sampled_frame_numbers.append(int(current_frame_num))

                current_frame_num += 1
        finally:
            cap.release()

        if not sampled_parts:
            return {
                "releve_maintained_majority_estimate": None,
                "confidence": 0.0,
                "notes": "Failed to extract any frames for relevé review.",
                "flat_evidence_frame_numbers": [],
                "frame_statuses": [],
                "used_model": None,
                "used_frames": {"hold_start_frame": start_frame, "hold_end_frame": end_frame, "sampled": 0},
            }

        prompt = (
            "You are an expert judge reviewing RELEVÉ (tip-toe / heel lifted) during a Penche HOLD.\n"
            "You will receive frames ONLY from the hold (and preferably from the hands-off part of the hold).\n"
            "Do NOT judge feet outside the provided review window.\n\n"
            "Images are in repeating triplets for the same frame number:\n"
            "1) FULL frame\n"
            "2) FEET-L zoom\n"
            "3) FEET-R zoom\n\n"
            "Task per sampled frame number: determine whether the SUPPORT FOOT is in RELEVÉ.\n"
            "Definition:\n"
            "- RELEVE: heel clearly lifted off the floor (weight on ball/toe), heel NOT contacting ground.\n"
            "- FLAT: heel is on the floor / clearly not tip-toe.\n"
            "- UNCLEAR: feet/heel not visible enough to decide.\n\n"
            "Be conservative:\n"
            "- Do NOT guess from shadows or proximity.\n"
            "- If you cannot clearly see the heel-ground relationship, output UNCLEAR.\n\n"
            f"Hold window frames: {start_frame}..{end_frame}\n"
            f"Review window frames: {review_start}..{review_end}\n"
            f"Sampled frame numbers (chronological): {sampled_frame_numbers}\n"
            f"IMPORTANT: frame_statuses MUST have exactly {len(sampled_frame_numbers)} items, in the same order.\n"
            "Return ONLY JSON matching the schema.\n"
        )

        schema = {
            "type": "OBJECT",
            "required": ["frame_statuses", "confidence", "notes"],
            "properties": {
                "frame_statuses": {
                    "type": "ARRAY",
                    "items": {"type": "STRING", "enum": ["RELEVE", "FLAT", "UNCLEAR"]},
                },
                "confidence": {"type": "NUMBER", "description": "0.0 to 1.0"},
                "notes": {"type": "STRING"},
            },
        }

        last_error: Exception | None = None
        model_try_order = ["gemini-2.5-pro", *self._model_names_to_try()]
        seen_models: set[str] = set()
        for model_name in model_try_order:
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=[types.Part.from_text(text=prompt), *sampled_parts],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                if parsed is None:
                    parsed = json.loads(response.text or "{}")

                statuses = list(parsed.get("frame_statuses") or [])
                if len(statuses) < len(sampled_frame_numbers):
                    statuses.extend(["UNCLEAR"] * (len(sampled_frame_numbers) - len(statuses)))
                if len(statuses) > len(sampled_frame_numbers):
                    statuses = statuses[: len(sampled_frame_numbers)]

                confidence = float(parsed.get("confidence") or 0.0)
                notes = str(parsed.get("notes") or "")

                # Decide majority, ignoring UNCLEAR.
                known = [s for s in statuses if s in ("RELEVE", "FLAT")]
                releve_majority: bool | None
                if len(known) < 3:
                    # If the model didn't provide usable per-frame labels, be conservative.
                    # However, if its notes explicitly say "flat" or "relevé" clearly, we can
                    # use that as a weak fallback instead of returning null.
                    notes_l = notes.lower()
                    if "flat" in notes_l and "unclear" not in notes_l:
                        releve_majority = False
                        confidence = min(confidence, 0.7)
                    elif "relev" in notes_l and "unclear" not in notes_l:
                        releve_majority = True
                        confidence = min(confidence, 0.7)
                    else:
                        releve_majority = None
                        confidence = min(confidence, 0.4)
                        if not notes:
                            notes = "Insufficient per-frame releve labels returned."
                        else:
                            notes = notes + " (Insufficient per-frame releve labels.)"
                else:
                    releve_count = sum(1 for s in known if s == "RELEVE")
                    flat_count = sum(1 for s in known if s == "FLAT")
                    releve_majority = releve_count > flat_count

                flat_frames = [int(f) for f, st in zip(sampled_frame_numbers, statuses) if st == "FLAT"]

                return {
                    "releve_maintained_majority_estimate": releve_majority,
                    "confidence": confidence,
                    "notes": notes,
                    "flat_evidence_frame_numbers": flat_frames,
                    "frame_statuses": statuses,
                    "used_model": model_name,
                    "used_frames": {
                        "hold_start_frame": int(start_frame),
                        "hold_end_frame": int(end_frame),
                        "review_start_frame": int(review_start),
                        "review_end_frame": int(review_end),
                        "sampled": int(len(sampled_frame_numbers)),
                    },
                }
            except Exception as e:
                last_error = e
                if "404" not in str(e) and "not found" not in str(e).lower():
                    raise
                continue

        return {
            "releve_maintained_majority_estimate": None,
            "confidence": 0.0,
            "notes": f"Could not find a valid Gemini model for relevé review. Last error: {last_error}",
            "flat_evidence_frame_numbers": [],
            "frame_statuses": [],
            "used_model": None,
            "used_frames": {
                "hold_start_frame": int(start_frame),
                "hold_end_frame": int(end_frame),
                "sampled": int(len(sampled_frame_numbers)),
            },
        }
    
    def _sample_report_frames(
        self,
        video_path: Optional[str] = None,
        peak_image_path: Optional[str] = None,
        hold_window_1s: Optional[Dict[str, Any]] = None,
        max_frames: int = 5,
    ) -> List[types.Part]:
        """
        Sample key frames for the comprehensive report vision input.
        Returns list of image Parts: from peak_image_path and/or video (hold window or middle).
        """
        parts: List[types.Part] = []
        if peak_image_path and os.path.isfile(peak_image_path):
            try:
                with open(peak_image_path, "rb") as f:
                    data = f.read()
                parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
            except Exception:
                pass
        if video_path and os.path.isfile(video_path) and len(parts) < max_frames:
            try:
                import cv2
            except Exception:
                return parts
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return parts
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                start_f, end_f, peak_f = None, None, None
                if hold_window_1s:
                    start_f = hold_window_1s.get("window_start_frame")
                    end_f = hold_window_1s.get("window_end_frame")
                    peak_f = hold_window_1s.get("peak_frame_within_hold")
                if isinstance(start_f, int) and isinstance(end_f, int) and end_f >= start_f:
                    indices = []
                    peak_f = int(peak_f) if isinstance(peak_f, (int, float)) else (start_f + end_f) // 2
                    for i in [start_f, (start_f + peak_f) // 2, peak_f, (peak_f + end_f) // 2, end_f]:
                        if start_f <= i <= end_f and i not in indices:
                            indices.append(i)
                    if len(indices) > max_frames - len(parts):
                        indices = indices[: max_frames - len(parts)]
                    for frame_idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        h, w = frame.shape[:2]
                        if w > 960:
                            scale = 960 / float(w)
                            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                        if ok:
                            parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
                else:
                    need = max_frames - len(parts)
                    step = max(1, total_frames // (need + 1)) if total_frames > 0 else 1
                    for i in range(need):
                        frame_idx = min(i * step, total_frames - 1) if total_frames else 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                        ret, frame = cap.read()
                        if not ret:
                            break
                        h, w = frame.shape[:2]
                        if w > 960:
                            scale = 960 / float(w)
                            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                        if ok:
                            parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
            finally:
                cap.release()
        return parts

    def _sample_overall_report_frames(
        self,
        video_paths: List[str],
        max_frames_per_video: int = 3,
        max_total_frames: int = 15,
    ) -> List[types.Part]:
        """
        Sample key frames from each video for the overall person report.
        Returns list of Parts: for each video, a short text label then that video's image Parts.
        """
        parts: List[types.Part] = []
        total = 0
        try:
            import cv2
        except Exception:
            return parts
        per_video = min(max_frames_per_video, max(1, max_total_frames // max(1, len(video_paths))))
        for idx, video_path in enumerate(video_paths):
            if total >= max_total_frames:
                break
            if not video_path or not os.path.isfile(video_path):
                continue
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                need = min(per_video, max_total_frames - total)
                if need <= 0 or total_frames <= 0:
                    continue
                parts.append(types.Part.from_text(text=f"【动作 {idx + 1} 关键帧】"))
                step = max(1, total_frames // (need + 1))
                for i in range(need):
                    frame_idx = min(i * step, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    if w > 960:
                        scale = 960 / float(w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ok:
                        parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
                        total += 1
            finally:
                cap.release()
        return parts

    async def _generate_content(self, user_prompt: str) -> str:
        """Call Gemini with retry over model names. Returns response.text."""
        model_names_to_try = self._model_names_to_try()
        last_error = None
        for model_name in model_names_to_try:
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=user_prompt
                )
                return response.text
            except Exception as e:
                last_error = e
                if "404" not in str(e) and "not found" not in str(e).lower():
                    raise
                continue
        raise RuntimeError(
            f"Could not find a valid Gemini model. Tried: {model_names_to_try}. "
            f"Last error: {last_error}. "
            f"Please check your API key and available models."
        ) from last_error

    async def _generate_content_with_vision(self, user_prompt: str, image_parts: List[types.Part]) -> str:
        """Call Gemini with text + images; prefers vision-capable models. Returns response.text."""
        model_try_order = ["gemini-2.5-pro", "gemini-2.5-flash", *self._model_names_to_try()]
        seen = set()
        last_error = None
        contents = [types.Part.from_text(text=user_prompt), *image_parts]
        for model_name in model_try_order:
            if model_name in seen:
                continue
            seen.add(model_name)
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                return response.text or ""
            except Exception as e:
                last_error = e
                if "404" not in str(e) and "not found" not in str(e).lower():
                    raise
                continue
        raise RuntimeError(
            f"Could not find a valid Gemini vision model. Last error: {last_error}"
        ) from last_error

    async def chat_about_report(
        self,
        report_context: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Answer user questions about a gymnastics report using the same Gemini model.
        report_context: full report text (comprehensive or overall).
        user_message: current user question.
        history: optional list of {"role": "user"|"model", "content": "..."} for multi-turn.
        """
        if not (report_context or "").strip():
            return "暂无报告内容可供参考，请先完成分析并查看报告。"
        if not (user_message or "").strip():
            return "请输入您的问题。"

        lines = [
            "你是一位专业的体操教练与裁判助手。请回答用户问题。",
            "回答要简洁、专业、有针对性。可用中文。若报告未涉及所问内容，请如实说明。",
            "",
            "========== 报告内容 ==========",
            (report_context or "").strip(),
            "==================================",
            "",
        ]
        history = history or []
        for turn in history:
            role = turn.get("role") or "user"
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}：{content}")
            lines.append("")
        lines.append("用户：" + (user_message or "").strip())
        prompt = "\n".join(lines)
        return await self._generate_content(prompt)

    async def evaluate(self, tool_output: Dict[str, Any], context_prompt: str) -> str:
        """
        Send tool output to LLM for evaluation
        """
        import json
        full_prompt = f"""
        {context_prompt}
        
        Here is the technical data from the computer vision analysis:
        {json.dumps(tool_output, indent=2)}
        
        Please provide your judging assessment based on the rules provided.
        """
        return await self._generate_content(full_prompt)

    async def simple_move_report(
        self,
        tool_name: str,
        tool_output: Dict[str, Any],
        judge_verdict: Optional[str] = None,
    ) -> str:
        """
        Generate a short move report in Chinese (~100 words): D score, E score,
        得分点/扣分点 in words, and one-sentence summary. Used for all three tools.
        """
        import json
        data_blob = json.dumps(tool_output, indent=2, ensure_ascii=False)
        verdict_blob = f"\n\n裁判详细结论（供参考）：\n{judge_verdict}" if judge_verdict else ""
        prompt = f"""你是一位体操裁判。请根据以下技术数据，用中文写一份「简易动作报告」（约100字）。

System Role: 你是一名资深的艺术体操国际级裁判，请严格依据 2025-2028 FIG 规则及提供的 CV 测量数据生成简易报告
Input Data: 接收 Computed Metrics 中的所有数值（如 split_angle_deg, hold_seconds, releve_maintained 等）
Output Format (严格遵守以下顺序及逻辑):
1. 动作名称：动作名称及编号（例如：Penche 俯平衡 - 2.1106）
2. 难度判定 (D分) 与完成审计 (E分)
- D分：判定 (有效/无效)，得分 (数值)
- E分：总扣分 (数值)
3. 得分点（D）与扣分点（E）
- 要求：每一项得分与扣分点，通过实测数据进行对照打分。
4. 总结：
- 要求：1-2句话简短总结该次动作表现，请基于得分（D）与失分（E）情况进行梯队等级评价——需进步/已达标/表现佳/很完美
（示例：本次动作技术规范，表现佳）

技术数据（JSON）：
{data_blob}
{verdict_blob}

请直接输出中文报告，不要输出 JSON 或代码块。"""
        return await self._generate_content(prompt)

    async def comprehensive_report(
        self,
        tool_name: str,
        tool_output: Dict[str, Any],
        judge_verdict: Optional[str] = None,
        *,
        video_path: Optional[str] = None,
        peak_image_path: Optional[str] = None,
        hold_window_1s: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a student-oriented comprehensive report in Chinese (~200 words):
        well-done points, improvement points, recommended next steps.
        If video_path and/or peak_image_path / hold_window_1s are provided, key frames
        are sent to the LLM so the report can be informed by both statistics and vision
        (still primarily based on the technical data).
        """
        import json
        data_blob = json.dumps(tool_output, indent=2, ensure_ascii=False)
        verdict_blob = f"\n\n裁判详细结论（供参考）：\n{judge_verdict}" if judge_verdict else ""
        prompt = f"""你是一名资深的艺术体操国际级裁判及高级教练。请基于提供的 Computed Metrics 数据与视频视觉观察，输出一份针对运动员的深度分析报告（约200字）。

动作/工具名称：{tool_name}

[Output Principles]:
- 数据优先原则：始终以 JSON 格式的 Computed Metrics 实测数据作为判定的事实准则；严禁凭多模态视觉直觉推翻、修正或得出任何与实测数据相悖的结论。
- 观察任务：请开启多模态审计权限，在参考数据的同时深度观察视频，作为判定的视觉辅助。
- 数据驱动：分析必须与实测数据（如180度开度、定格秒数、结环角度）形成强闭环。

Output Format（严格遵守，勿用 markdown 表格）：
- 用流畅的中文写作，像一个中文是母语的人一样。
- 请用**分点叙述**（小标题 + 短段落或 bullet），不要使用 markdown 表格（禁止使用 | 分隔符），不要在行末或句中加 |。
- 四维分析每项单独成段，每段先写维度名再写内容，保持段落简洁易读。

1. 身体难度 (DB) 四维深度分析
- 姿态 (Shape)：先写「数据判定」（劈叉开度、结环偏差等），再写「视觉观察」一两句。
- 动力性 (Dynamics)：数据判定（如转体圈数）；视觉观察（起势、滞空感、速度感等）一两句。
- 稳定性 (Stability)：数据判定（定格时间、重心等）；视觉观察（轴心、晃动等）一两句。
- 动作规范性 (Precision)：视觉观察（立踵、脚背、膝盖、姿态规范性）一两句。

2. 身体难度动作优点与不足
- 核心优点：引用实测数据说明突出能力，一两句即可。
- 关键短板：引用失分最重的指标说明不足，一两句即可。

3. 针对性练习建议
- 根据关键短板给出 1～2 条具体练习建议，每条一两句。

**重要**：结论必须以技术数据（JSON）和裁判结论为主要依据。若下方附有关键动作画面，可结合画面补充观察，但不要与数据矛盾。整篇请用自然段落与分点，勿用表格或 | 符号。

技术数据（JSON）：
{data_blob}
{verdict_blob}

请直接输出中文综合报告，不要输出 JSON 或代码块。"""
        image_parts = self._sample_report_frames(
            video_path=video_path,
            peak_image_path=peak_image_path,
            hold_window_1s=hold_window_1s,
            max_frames=5,
        )
        if image_parts:
            prompt_vision = prompt + "\n\n下方附有关键动作画面（按时间或动作阶段），供你结合数据参考。"
            return await self._generate_content_with_vision(prompt_vision, image_parts)
        return await self._generate_content(prompt)

    async def overall_person_report(
        self,
        per_element_results: list,
        video_paths: Optional[List[str]] = None,
    ) -> dict:
        """
        Generate 反馈报告形式3：身体难度与运动员深度分析报告.
        per_element_results: list of dicts with keys tool_name, verdict (optional), simple_report, comprehensive_report.
        video_paths: optional list of video file paths (same order as per_element_results); when provided, key frames are sent so the LLM can use vision in addition to the text data.
        Returns dict with: per_element (pro/con/comment per element), radar (A,B,C,D 1-10), athlete_profile, practice_guide, full_report.
        """
        import json
        from google.genai import types

        elements_blob = json.dumps(
            [
                {
                    "tool_name": r.get("tool_name", ""),
                    "verdict": r.get("verdict"),
                    "simple_report": r.get("simple_report"),
                    "comprehensive_report": r.get("comprehensive_report"),
                }
                for r in per_element_results
            ],
            indent=2,
            ensure_ascii=False,
        )

        prompt = """你是一名资深的艺术体操国际级裁判及高级教练。请根据 Computed Metrics 实测数据 与视频多模态观察，生成一份具备因果诊断逻辑的运动员深度分析报告。

[Output Principles]:
- 硬数据 (Source of Truth)：Computed Metrics 中的数值（如角度、时长、位移、立踵状态）是判定优缺点的绝对依据。
- 多模态观察 (Visual Assist)：利用视觉能力观察数据无法覆盖的细节，如：进入动作的动力性，跳跃滞空感，手部隐蔽支撑，身体出现细微晃动/抖动。

Input Data：
- 每个身体难度（平衡/跳跃/转体等）已有：简易报告、综合报告，以及裁判结论，请根据此综合写出整体报告。

Output format（必须输出的 JSON 结构）：
- 每个身体难度动作：数组，与输入顺序一致。每项包含：
    - 动作名称（编号）
    - 得分 (D) 与扣分 (E) 情况

- 运动员全方位能力评估：四维度图像与文字总结，每项 1–10 分（整数）
    - 评分逻辑：根据各类身体难度 (DB) 的四维深度分析，综合评估该运动员的各项核心能力：
        - 姿态 (Shape)：
            - 数据判定：劈叉开度，结环偏差。
        - 动力性 (Dynamics)：
            - 数据判定：转体动作圈数
            - 视觉观察：所有类型动作的动力过程（例如：起势果断，力量传递流畅；或进入动作过程存在借力，不够干脆）
            - 视觉观察：平衡动作的进入和撤势速度，跳跃动作的弹跳高度和滞空感，转体动作的速度感。
        - 稳定性 (Stability)：
            - 数据判定：定格时间，失去重心，地面接触。
            - 视觉观察：重心位移，轴心不稳定，动作过程中存在明显的晃动和移动。
        - 动作规范性 (Precision)：
            - 视觉观察：立踵高度，脚背弧度，膝盖锁死，固定姿态规范性。

- 运动员画像：根据“全方位能力画像”和“四维分析”
    - 身体难度倾向分析：对比并锁定该运动员最擅长和最薄弱的身体难度类别。
    - 核心优势：锁定得分最高的难度动作，对应运动员最突出的底层能力，给予权威级的肯定和表扬。
    - 核心弱项：锁定得分最低，且在多个动作中共同出现的底层能力缺陷。

- 练习建议与训练指南：
    - [针对性训练]：针对上述短板，提供专业艺术体操练习方法（如：针对立踵不稳，建议加强踝关节力量训练及居家提踵练习）
    - [日常重点]：指明下一次训练中应优先关注的 1-2 个核心技术点。
    
- 总结：约 50 字的全文总结，精炼要点与优缺点，陈述改进方向与总体评价；语言风格保持专业和力量感，确保家长能听懂价值，学生能获得动力。

请仅输出合法 JSON，不要输出 markdown 代码块或其它前后文字。"""

        full_prompt = f"{prompt}\n\n## 各动作分析结果\n{elements_blob}"
        if video_paths and len(video_paths) == len(per_element_results):
            full_prompt += "\n\n下方附有各动作的关键帧（与上面输入顺序一致：动作1、动作2…），供你结合视觉观察完善四维画像与运动员画像。结论仍以数据为主要依据，视觉作为辅助。"
        frame_parts: List[types.Part] = []
        if video_paths and len(video_paths) == len(per_element_results):
            frame_parts = self._sample_overall_report_frames(
                video_paths,
                max_frames_per_video=3,
                max_total_frames=15,
            )

        schema = {
            "type": "OBJECT",
            "required": ["per_element", "radar", "athlete_profile", "practice_guide", "full_report"],
            "properties": {
                "per_element": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "required": ["element_name", "pro", "con", "comment"],
                        "properties": {
                            "element_name": {"type": "STRING"},
                            "pro": {"type": "STRING"},
                            "con": {"type": "STRING"},
                            "comment": {"type": "STRING"},
                        },
                    },
                },
                "radar": {
                    "type": "OBJECT",
                    "required": ["A", "B", "C", "D"],
                    "properties": {
                        "A": {"type": "INTEGER", "description": "1-10"},
                        "B": {"type": "INTEGER", "description": "1-10"},
                        "C": {"type": "INTEGER", "description": "1-10"},
                        "D": {"type": "INTEGER", "description": "1-10"},
                    },
                },
                "athlete_profile": {"type": "STRING"},
                "practice_guide": {"type": "STRING"},
                "full_report": {"type": "STRING"},
            },
        }

        contents: List[types.Part] = [types.Part.from_text(text=full_prompt)]
        if frame_parts:
            contents.extend(frame_parts)
        model_try_order = ["gemini-2.5-pro", "gemini-2.5-flash", *self._model_names_to_try()]
        last_error = None
        for model_name in model_try_order:
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                if parsed is None:
                    parsed = json.loads(response.text or "{}")
                r = parsed.get("radar") or {}
                for k in ("A", "B", "C", "D"):
                    if k in r and isinstance(r[k], (int, float)):
                        r[k] = max(1, min(10, int(r[k])))
                parsed["radar"] = r
                return parsed
            except Exception as e:
                last_error = e
                if "404" not in str(e) and "not found" not in str(e).lower():
                    raise
                continue
        raise RuntimeError(
            f"Could not generate overall report. Last error: {last_error}"
        ) from last_error
