#============================================================================================================
#============================================================================================================
# Inputs: method, target, (source), split_type, splitSeed, k_fn modelSeed
def outer_run_model():
    # get args, including reading in k from k_fn!! = TODO
    # get target_train_df by splitting on split_seed
    source_col, target_col = get_column_names(method, source_name, target_name)
    target_train_df, target_test_df, n_samp, n_drug = helpers.get_target(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type)
    # run model using model_seed, target_train_df, target_test_df, k
    s_idx_val, d_idx_val = helpers.get_sample_drug_indices(target_test_df)
    #print('Run ' + str(N_MODELS) + ' model restarts')
    if method == 'target_only':
        train_predict, val_predict = run_predict_target_only(model_seed, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k)
    elif method == 'transfer':
        source_df = helpers.get_source(data_fn, source_col)
        train_predict, val_predict = run_predict_transfer(model_seed, source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, 
            n_drug, n_steps, k)
    train_corr = evaluate_correlation(train_predict, train_df, target_col)
    val_corr = evaluate_correlation(val_predict, val_df, target_col)
    # save outputs!
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(val_corr, write_dir + '/val.pkl')
