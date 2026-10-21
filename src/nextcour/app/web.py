"""The web page (roadmap Phase 1): a username in, the Planning list out, ranked and explained.

- GET  /                                  a form for the username
- GET  /u/{user}                          the ranked Planning list, held-back shows and picks
- GET  /u/{user}/status                   whether explanations are still being written (polled)
- POST /u/{user}/write                    write the missing explanations now
- POST /u/{user}/not-interested/{id}      the "not interested" button; /interested/{id} undoes it
- POST /u/{user}/pick/{id}                "I'm watching this": logs the pick with its rank and explanation
- GET  /api/u/{user}/planning             the same list as JSON
- GET  /compare, POST /compare/{comparison}/{id}/{choice}, GET /compare/results
                                          the blind comparison of the two writer models

Pages are rendered on the server with Jinja2, and HTMX handles the buttons, so there's no
JavaScript build step (design note, Implementation defaults). Missing explanations are written
in a background thread while the page polls; a template stands in until they're ready.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from nextcour.app import service
from nextcour.app.feedback import FeedbackStore, Pick
from nextcour.app.service import Planner, Ranking
from nextcour.explain import compare
from nextcour.explain.writer import TEMPLATE, Explanation, Writer
from nextcour.ingest import anilist

TEMPLATES = Path(__file__).parent / "templates"
RETRY_AFTER = 600.0  # after a failed run, wait this long before writing again on its own


@dataclass
class Jobs:
    """Background explanation writing, one job per user at a time."""

    clock: Callable[[], float] = time.time
    running: dict[str, threading.Thread] = field(default_factory=dict)
    finished: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def busy(self, user: str) -> bool:
        thread = self.running.get(user.lower())
        return thread is not None and thread.is_alive()

    def recently_finished(self, user: str) -> bool:
        return self.clock() - self.finished.get(user.lower(), float("-inf")) < RETRY_AFTER

    def start(self, user: str, work: Callable[[], str | None]) -> bool:
        """Run `work` in the background; what it returns, if anything, is shown as a problem."""
        key = user.lower()
        with self.lock:
            if self.busy(key):
                return False

            def run() -> None:
                try:
                    problem = work()
                    if problem:
                        self.errors[key] = str(problem)
                    else:
                        self.errors.pop(key, None)
                except Exception as error:  # the page reports it; the thread mustn't die silently
                    self.errors[key] = str(error)
                finally:
                    self.finished[key] = self.clock()

            thread = threading.Thread(target=run, name=f"explain-{key}", daemon=True)
            self.running[key] = thread
            thread.start()
            return True


def create_app(
    planner: Planner,
    writer: Writer,
    feedback: FeedbackStore,
    votes: compare.Votes,
    comparisons: Path | None = None,
    jobs: Jobs | None = None,
) -> FastAPI:
    app = FastAPI(title="NextCour", description="Explainable anime recommendations for AniList")
    templates = Jinja2Templates(directory=str(TEMPLATES))
    jobs = jobs or Jobs()

    def ranking_or_error(user: str) -> Ranking:
        try:
            return planner.ranking(user)
        except anilist.NotFound as error:
            raise HTTPException(404, f"No public AniList anime list for {user}.") from error
        except anilist.AniListError as error:
            # AniList suspends its API during severe outages (a 403); lists can't load until it's back.
            raise HTTPException(
                503,
                f"AniList's API isn't available right now, so the list can't load. Try again later. "
                f"AniList said: {error}. The writer comparison at /compare still works.",
            ) from error

    def page_data(ranking: Ranking) -> dict[str, Any]:
        found = service.explanations(writer, ranking)
        waiting = sum(1 for e in found.values() if _not_written(e))
        return {
            "ranking": ranking,
            "explanations": found,
            "waiting": waiting,
            "writing": jobs.busy(ranking.user),
            "error": jobs.errors.get(ranking.user.lower()),
            "can_write": writer.client is not None,
            "picks": feedback.picks(ranking.user),
        }

    def start_writing(ranking: Ranking) -> bool:
        if writer.client is None:
            return False
        return jobs.start(ranking.user, lambda: service.failures(service.write_missing(writer, ranking)))

    @app.exception_handler(HTTPException)
    def error_page(request: Request, error: HTTPException) -> Response:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": error.detail}, status_code=error.status_code)
        return templates.TemplateResponse(
            request, "error.html", {"message": error.detail}, status_code=error.status_code
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, user: str | None = None) -> Response:
        if user and user.strip():
            return RedirectResponse(f"/u/{quote(user.strip(), safe='')}", status_code=303)
        return templates.TemplateResponse(request, "index.html", {})

    @app.get("/u/{user}", response_class=HTMLResponse)
    def planning(request: Request, user: str) -> Response:
        ranking = ranking_or_error(user)
        data = page_data(ranking)
        if data["waiting"] and not data["writing"] and not jobs.recently_finished(ranking.user):
            data["writing"] = start_writing(ranking)
        return templates.TemplateResponse(request, "planning.html", data)

    @app.get("/u/{user}/status", response_class=HTMLResponse)
    def status(request: Request, user: str) -> Response:
        if jobs.busy(user):
            return templates.TemplateResponse(request, "_status.html", {"user": user, "writing": True})
        return Response(headers={"HX-Refresh": "true"})

    @app.post("/u/{user}/write", response_class=HTMLResponse)
    def write(request: Request, user: str) -> Response:
        start_writing(ranking_or_error(user))
        return templates.TemplateResponse(request, "_status.html", {"user": user, "writing": jobs.busy(user)})

    @app.post("/u/{user}/not-interested/{anilist_id}")
    def not_interested(user: str, anilist_id: int) -> Response:
        feedback.mark_not_interested(user, anilist_id)
        return Response(headers={"HX-Refresh": "true"})

    @app.post("/u/{user}/interested/{anilist_id}")
    def interested(user: str, anilist_id: int) -> Response:
        feedback.undo_not_interested(user, anilist_id)
        return Response(headers={"HX-Refresh": "true"})

    @app.post("/u/{user}/pick/{anilist_id}", response_class=HTMLResponse)
    def pick(user: str, anilist_id: int) -> Response:
        ranking = ranking_or_error(user)
        row = next((r for r in ranking.rows if r.show.anilist_id == anilist_id), None)
        if row is None:
            raise HTTPException(404, "That show isn't on the ranked list.")
        explanation = service.explanations(writer, ranking)[anilist_id]
        feedback.add_pick(
            Pick(
                ranking.user,
                anilist_id,
                row.show.title,
                row.rank,
                row.score.predicted,
                explanation.text,
                feedback.now(),
            )
        )
        return HTMLResponse('<span class="picked">Noted: you picked this.</span>')

    @app.get("/api/u/{user}/planning")
    def planning_json(user: str) -> dict[str, Any]:
        ranking = ranking_or_error(user)
        found = service.explanations(writer, ranking)
        return {
            "user": ranking.user,
            "history_size": ranking.history_size,
            "ranked": [
                {
                    "rank": row.rank,
                    "anilist_id": row.show.anilist_id,
                    "title": row.show.title,
                    "predicted_enjoyment": round(row.score.predicted, 4),
                    "crowd_prior": round(row.score.crowd_prior, 4),
                    "taste_match": round(row.score.taste, 4),
                    "crowd_links": round(row.score.links, 4),
                    "rank_by_average_score": row.average_rank,
                    "average_score": row.show.average_score,
                    "watch_minutes": row.show.watch_minutes,
                    "series_watched": list(row.series),
                    "explanation": found[row.show.anilist_id].as_dict(),
                    "evidence": [
                        {"id": e.id, "kind": e.kind, "fact": e.fact, "effect": e.described_effect}
                        for e in row.item.evidence
                    ],
                }
                for row in ranking.rows
            ],
            "held_back": [
                {"anilist_id": h.show.anilist_id, "title": h.show.title, "waiting_on": list(h.waiting_on)}
                for h in ranking.held
            ],
            "not_interested": [{"anilist_id": s.anilist_id, "title": s.title} for s in ranking.hidden],
        }

    # --- The blind comparison ---------------------------------------------------------------

    def current_comparison() -> dict[str, Any]:
        comparison = compare.latest(comparisons)
        if comparison is None:
            raise HTTPException(
                404, "No comparison yet. Run `uv run nextcour compare-writers --user <name>`."
            )
        return comparison

    @app.get("/compare", response_class=HTMLResponse)
    def comparison_page(request: Request) -> Response:
        comparison = current_comparison()
        chosen = votes.of(comparison["id"])
        return templates.TemplateResponse(
            request,
            "compare.html",
            {"comparison": comparison, "votes": chosen, "choices": compare.CHOICES},
        )

    @app.post("/compare/{comparison_id}/{anilist_id}/{choice}", response_class=HTMLResponse)
    def vote(request: Request, comparison_id: str, anilist_id: int, choice: str) -> Response:
        comparison = current_comparison()
        if comparison_id != comparison["id"] or choice not in compare.CHOICES:
            raise HTTPException(400, "That vote doesn't match the current comparison.")
        votes.vote(comparison_id, anilist_id, choice)
        chosen = votes.of(comparison_id)
        return templates.TemplateResponse(
            request,
            "_vote.html",
            {
                "comparison": comparison,
                "item": next(i for i in comparison["items"] if i["anilist_id"] == anilist_id),
                "votes": chosen,
                "choices": compare.CHOICES,
                "oob_progress": True,
            },
        )

    @app.get("/compare/results", response_class=HTMLResponse)
    def comparison_results(request: Request, reveal: bool = False) -> Response:
        comparison = current_comparison()
        chosen = votes.of(comparison["id"])
        return templates.TemplateResponse(
            request,
            "compare_results.html",
            {
                "comparison": comparison,
                "votes": chosen,
                "choices": compare.CHOICES,
                "tallies": compare.tally(comparison, chosen),
                # Which model was A or B stays hidden until every show is voted, so it can't sway a vote.
                "revealed": reveal or len(chosen) >= len(comparison["items"]),
            },
        )

    return app


def _not_written(explanation: Explanation) -> bool:
    return explanation.writer == TEMPLATE and explanation.note == "not written yet"
