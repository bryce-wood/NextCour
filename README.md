# NextCour

**NextCour: explainable anime recommendations for AniList.**

NextCour is an anime recommender being built on top of AniList. It will read a public AniList anime list, learn what that person actually enjoys (relative scores, drops, binge speed, rewatches), and recommend anime from the whole catalog, with an explanation for every pick that traces back to real evidence. It's a personal, non-commercial project and isn't affiliated with AniList or MyAnimeList.

**Status:** in development. The first useful version works: it ranks the shows on someone's AniList Planning list with a hand-weighted scorer, keeps sequels in watch order, and explains each pick. Trained models come next.

## Setup

Needs [uv](https://docs.astral.sh/uv/). uv installs the right Python version itself.

```bash
uv sync
uv run pytest
```

Checks that CI also runs:

```bash
uv run ruff check
uv run ruff format --check
```

A few tests use a personal, private test set (the author's frozen "works for me" shows) and skip when it isn't there, as in CI.

## Use it

It needs the catalog and franchise groups first (see Data below):

```bash
uv run nextcour catalog
uv run nextcour franchises
```

Then, for any public AniList user:

```bash
uv run nextcour serve
```

That runs the web page at http://127.0.0.1:8000/, on your computer only. Enter a user name to see their Planning list ranked, with a short explanation for each show. The same list is at `/api/u/<name>/planning` as JSON. On the command line, `uv run nextcour rank --user <name>` prints the ranking and `uv run nextcour explain --user <name>` prints the explanations.

Explanations are written by Google's Gemini on its free tier when `GEMINI_API_KEY` is set in `.env` (copy `.env.example`). Without a key, each show gets a plain sentence built from the same facts. Every sentence has to cite the facts it uses, and a check drops any that don't hold up.

## Evaluate

The models are measured on MAL 2018 users (convert the datasets first; see Data below):

```bash
uv run nextcour split
uv run nextcour calibrate
uv run nextcour eval baselines
```

`split` divides MAL 2018 by time at 2017-07-01: each user's entries before then are the input, and the shows they watched after it are what the models try to predict. Users are divided into train, dev and test groups. `calibrate` measures the labels for unscored entries on train users. `eval baselines` scores the baselines on dev users with 95% confidence intervals and writes a results table to `data/results/`.

## Data

Datasets live in `data/`, which git ignores. Nothing in `data/` is ever committed or redistributed. [`manifests/datasets.json`](manifests/datasets.json) records each file's source, size and SHA-256 checksum, so every result can name the exact data it used.

| Data | Where it goes | How to get it |
| --- | --- | --- |
| anime-offline-database | `data/raw/anime-offline-database/<release>/` | `uv run nextcour catalog` downloads it |
| MAL 2018 | `data/raw/mal-2018/` | Download from Kaggle (link below) and unzip |
| MAL 2020 | `data/raw/mal-2020/` | Download from Kaggle and unzip |
| MAL 2023 | `data/raw/mal-2023/` | Download from Kaggle and unzip |

Then convert the MAL datasets to Parquet once: `uv run nextcour convert-mal`.

## Data sources and credits

- **[anime-offline-database](https://github.com/manami-project/anime-offline-database)** by manami-project, under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/) and the Database Contents License (DbCL) v1.0. It provides the catalog and the AniList ↔ MyAnimeList ID mapping.
- **[MyAnimeList Dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist)** (2018) by azathoth42 on Kaggle, under the ODbL.
- **[Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)** by hernan4444 on Kaggle, under CC0.
- **[Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)** by dbdmobile on Kaggle, under the ODbL.
- **[AniList](https://anilist.co)**: show details and public lists come from the [AniList API](https://docs.anilist.co), used non-commercially under its terms and within its rate limit.

The three Kaggle datasets were collected from [MyAnimeList](https://myanimelist.net) by their authors. NextCour uses them only for offline training and evaluation, credits them here and in the app, and never redistributes them.
