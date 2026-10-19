# Immediate actions for Ahmed

These actions are required before the project can be reviewed as a runnable
production CV/MLOps system. They are ordered by blocking impact.

## P0 — Make a clean clone runnable

1. **Fix the clean-clone Docker build**

   The current `Dockerfile` contains:

   ```dockerfile
   COPY models ./models
   ```

   The repository clone has no `models/` directory, so
   `docker compose build` fails before the API starts.

   Choose one reproducible solution:

   - add a documented model-artifact bootstrap step that downloads a verified
     model before Docker builds;
   - track the required model artifacts with DVC and make `dvc pull` part of
     the documented setup; or
   - change the Docker build so the image can be built without a local
     `models/` directory, then fail clearly at startup if no model is present.

   The README must contain exact commands that work from a fresh clone.

2. **Make DVC actually reproducible**

   `dvc repro` currently fails because the clone contains `dvc.yaml` but is not
   initialized as a DVC repository. Add and commit the required DVC metadata
   and remote configuration, or document and test the exact initialization and
   pull procedure.

   The reviewer must be able to reproduce:

   - raw data download;
   - the train/validation/test manifest;
   - corruption generation;
   - model training;
   - optimization and drift reports.

3. **Fix the model-serving artifact path**

   `serving/app.py` computes `weights_path` and passes it to
   `PetBreedClassifier`, but `models/classifier.py` does not load
   `weights_path`.

   The serving code currently constructs a torchvision ResNet-50 with
   1,000 ImageNet outputs and maps ImageNet labels heuristically to the 37
   Oxford-IIIT Pet breeds. Training, however, creates a separate 37-class
   `PetClassifierBackbone` with a trained classification head.

   Make serving load the actual trained 37-class checkpoint and preserve the
   saved:

   - model architecture;
   - class map;
   - temperature-scaling value;
   - abstention threshold;
   - model version.

   Add an end-to-end test proving that the API uses the trained checkpoint and
   returns one of the 37 committed classes.

4. **Make the reported model metrics verifiable**

   The README reports 91.40% top-1 accuracy and optimization numbers, but the
   clean clone currently has no model files and cannot run the evaluation.
   Store the evaluation inputs, model artifact references, MLflow run IDs, and
   exact commands needed to reproduce or verify these values.

## P1 — Make CI pass

5. **Fix the current CI lint failure**

   The latest public CI run fails in `lint-and-format`. Running Ruff locally
   against `src` and `tests` reports 291 errors, including deprecated typing
   imports, import-order issues, unused imports, and formatting problems.

   Run Ruff and Black locally, fix the findings, and make the same commands
   pass in GitHub Actions.

6. **Install and test the real dependency set**

   The test suite requires FastAPI, PyTorch, MLflow, ONNX Runtime, and the
   optional optimization/serving/monitoring packages. Make the installation
   command deterministic and test it in CI from a clean Python 3.11
   environment.

   The final command must pass:

   ```bash
   pip install -e ".[dev,optimization,serving,monitoring]"
   pytest tests/ -v
   ```

7. **Make the Docker CI job meaningful**

   The Docker job should build the same artifact a reviewer can build. It must
   not depend on untracked local model files. Add a model bootstrap step or
   make the image build contract explicit and tested.

## P1 — Confirm the CV data contract

8. **Commit and validate the dataset contract**

   Verify that the committed labels and manifest contain exactly 37 classes,
   deterministic class indices, valid image paths, no duplicate image IDs, and
   disjoint train/validation/test splits.

9. **Make the corruption suite reproducible**

   Ensure every corruption severity is generated from the isolated test set,
   is tracked by DVC, and has a manifest linking each corrupted image to its
   clean source, corruption type, and severity.

10. **Prove train/serve preprocessing parity**

    Keep the existing `CanonicalPreprocessor` approach, but make the test run
    without downloading an untracked model. The test should verify RGB, RGBA,
    grayscale, and CMYK inputs and compare the actual trained model path used
    by the API with the offline evaluation path.

## P2 — Complete the production evidence

11. **Show real MLflow lineage**

    For each promoted model, provide the MLflow experiment, run, registered
    model version, artifact URI, data version, and model-card link. Do not
    rely only on hard-coded README benchmark tables.

12. **Verify optimization claims on the same hardware basis**

    The optimization table must identify which results were measured locally,
    on Colab, or on a T4. Include the exact command and input sample count
    for PyTorch, pruning, PTQ, QAT, ONNX, OpenVINO, distillation, and
    TensorRT/Triton.

13. **Demonstrate the closed loop**

    Provide one reproducible sequence showing:

    ```text
    corrupted batch
    -> drift detector
    -> Airflow/retraining trigger
    -> candidate evaluation
    -> quality gate
    -> MLflow registration or rejection
    -> canary/rollback decision
    ```

14. **Add a clean reviewer quickstart**

    The README currently contains a directory typo:

    ```text
    Pet Breed Classifiaction
    ```

    Replace it with commands that work exactly from the cloned repository.
    State clearly which steps are public and which require private GCP/DVC
    credentials.

## Evidence already observed by the reviewer

- `docker compose build` fails because `/models` is absent.
- `dvc repro` fails because the clone is not recognized as a DVC repository.
- Local test collection fails before dependencies are installed.
- Ruff reports 291 errors.
- The latest public CI run fails in `lint-and-format`.
- The repository contains strong intended design elements:
  canonical image preprocessing, typed FastAPI schemas, one-time startup
  model initialization, calibration, abstention, drift detectors, and a
  quality-gated retraining module.

Please address P0 items first and send back:

1. the new clean-clone commands;
2. a successful `docker compose build` and startup;
3. a successful `dvc repro` or exact public artifact bootstrap;
4. passing CI;
5. evidence that the API uses the trained 37-class checkpoint.
