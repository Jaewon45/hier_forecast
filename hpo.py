import optuna

def objective(trial):
    params = {
        'input_chunk_length': trial.suggest_categorical('input_chunk_length', [12, 24, 36]),
        'output_chunk_length': trial.suggest_categorical('output_chunk_length', [6, 12, 24]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'n_epochs': trial.suggest_int('n_epochs', 10, 50),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }

    model = NBEATSModel(**params,
                        pl_trainer_kwargs={
            'enable_progress_bar': False  # Reduce output clutter
        })

    try:
        model.fit(
            train,
            past_covariates=past_cov,
            epochs=params['n_epochs'],
            verbose=False
        )

        # Report intermediate score (enables pruning)
        pred = model.predict(len(val))
        mape_score = mape(val['EBIT'], pred)
        trial.report(mape_score, step=model.epochs_trained)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        return mape_score

    except Exception as e:
        print(f"Trial failed: {str(e)}")
        raise optuna.TrialPruned()



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)
#Best params: {'input_chunk_length': 36, 'output_chunk_length': 24, 'dropout': 0.11891699976631348, 'n_epochs': 27, 'batch_size': 128}