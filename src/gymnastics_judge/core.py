from typing import Protocol, Any, Dict, Optional
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

要求（必须全部用中文，且按此顺序）：
1. D分：说明有效或无效，以及难度分（0 或 0.5 等，视动作而定）。
2. E分：说明扣分情况（负数）。
3. 得分点与扣分点：用文字简要列出主要数据和要点，不要只堆数字。
4. 最后一句话总结该次动作表现。

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
    ) -> str:
        """
        Generate a student-oriented comprehensive report in Chinese (~200 words):
        well-done points, improvement points, recommended next steps.
        """
        import json
        data_blob = json.dumps(tool_output, indent=2, ensure_ascii=False)
        verdict_blob = f"\n\n裁判详细结论（供参考）：\n{judge_verdict}" if judge_verdict else ""
        prompt = f"""你是一位体操教练/裁判。请根据以下技术数据与裁判结论，用中文写一份「综合报告」（约200字），面向学生/运动员。

动作/工具名称：{tool_name}

要求（全部用中文）：
1. 做得好的地方：约3句话/要点。
2. 需要改进的地方：约3句话/要点。
3. 建议的下一步改进：约2～3句话/要点。

技术数据（JSON）：
{data_blob}
{verdict_blob}

请直接输出中文综合报告，不要输出 JSON 或代码块。"""
        return await self._generate_content(prompt)
