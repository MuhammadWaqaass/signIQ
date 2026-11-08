# Product Requirements Document (PRD)
**Product:** Signature Detection and Verification Service
**Status:** Draft v1
**Audience:** Team, instructors

---

## 1. Problem
Checking signatures on scanned documents is slow and manual. A reviewer must find each signature on the page and compare it by eye to a known reference. This is tedious, inconsistent between reviewers, and does not scale.

## 2. Goal
A web application where a user uploads a document image and a reference signature. The system finds every signature on the document, compares each one to the reference, and shows the results on the page with a similarity score and a recommended decision.

## 3. Users
| User | Need |
|---|---|
| Document reviewer | Quickly see where signatures are and how closely each matches a reference |
| Developer / integrator | A clean HTTP API to call from other systems |
| Course instructor | A working end-to-end system with proper architecture, queueing and deployment |

## 4. Scope

### In scope
- Detect signatures on an uploaded document image (object detection).
- Compare each detected signature to a reference signature (similarity score).
- Asynchronous processing through a message queue and a worker service.
- REST API (FastAPI) with job submission and job status.
- Web frontend that draws boxes and scores over the document.
- Free-tier cloud deployment, end to end.
- Automated training pipeline: new data on Hugging Face Datasets triggers remote training, tracked in Comet or MLflow, and the new model is deployed if better.
- AWS-style architecture diagram.

### Out of scope
- Legal or forensic certification of signatures.
- User accounts and authentication (possible later).
- PDF input (images only in v1).
- Real-time video.
- Training the detector with Ultralytics (AGPL license is not allowed in production).

## 5. User stories
1. As a reviewer, I upload a document and a reference signature so the system can check them.
2. As a reviewer, I see a boxed outline on every signature found, so I know where to look.
3. As a reviewer, I see a similarity score and an ACCEPT / REVIEW / REJECT verdict for each signature.
4. As a reviewer, I am not blocked while the model runs; I can see the job status.
5. As a developer, I submit a job and poll for the result through the API.
6. As a maintainer, uploading new labelled data automatically trains and ships a better model.

## 6. Functional requirements (summary)
Detailed requirements are in `SRS.md`.

| ID | Requirement | Priority |
|---|---|---|
| P1 | Upload document and reference images | Must |
| P2 | Detect signatures and return boxes with confidence | Must |
| P3 | Similarity score and verdict per detected signature | Must |
| P4 | Asynchronous job processing with status | Must |
| P5 | Frontend with box overlay and scores | Must |
| P6 | Deployed online on free tiers | Must |
| P7 | Auto-training pipeline with experiment tracking | Must |
| P8 | Architecture diagram (AWS style) | Must |
| P9 | Authentication and rate limiting | Could |
| P10 | PDF input | Could |

## 7. Success metrics
| Metric | Target |
|---|---|
| Job accepted by API | Response within 1 second |
| End-to-end result (CPU worker) | Within about 30 seconds for a single-page image (to be measured) |
| Detector quality | Meets the F1 / mAP recorded in `MODEL.md` |
| Similarity quality | Test accuracy 81.5%, ROC-AUC 0.902 (from the model card) |
| Demo | Full flow works on the deployed URL during the final presentation |

## 8. Constraints
- **License:** Training and serving code must use MIT, Apache or similarly permissive libraries. Ultralytics (AGPL) is not allowed in production. The deployed detector is trained with LibreYOLO.
- **Cost:** Free tiers only.
- **Time:** Daily checkpoints, final presentation on Tuesday.
- **Free-tier limits:** Redis command limits, sleeping services, and limited RAM shape the design (see Risks).

## 9. Key product decisions
- **Asynchronous by design.** Model inference is slow, so the API only queues work and returns a job ID.
- **Three-way verdict.** The similarity model is wrong in roughly 1 of 5 different-person pairs, so the product reports ACCEPT / REVIEW / REJECT and keeps a human in the loop for REVIEW.
- **Models live on the Hugging Face Hub.** The worker downloads them at startup, so a new model version is a new upload.

## 10. Milestones
| Day | Milestone |
|---|---|
| Day 1 | Planning, Jira board, architecture draft |
| Day 2 | Backend running locally end to end (API, queue, worker, both models) |
| Day 3 | Frontend, cloud storage, deployment; training pipeline started |
| Day 4 (Mon) | Training pipeline with tracking and auto-deploy |
| Tue | Final presentation |

## 11. Risks
| Risk | Impact | Mitigation |
|---|---|---|
| Redis free tier command limit reached | Queue stops | Disable result backend (done), test early, fall back to another Redis host |
| Free hosts sleep or restrict background workers | Jobs not processed | Add a health endpoint; consider a free VM for the worker |
| Similarity model weaker on phone photos | Poor scores in demo | Use scanned or flat images; state the limitation |
| Local file storage does not work across hosts | Deployment fails | Move storage to Supabase or R2 before deploying |
| Dataset licenses (CEDAR, BHSig260, HMS) | Compliance | State each source's terms in the presentation |

## 12. Open questions
- Detector Hub repository name (placeholder in use).
- Which tracking tool: Comet or MLflow.
- Which remote training platform runs the pipeline (HF Jobs, Kaggle, Colab-based or other).
