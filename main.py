import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data import (
    context_data_load,
    base_context_data_load,
    context_data_split,
    context_data_loader,
    load_cat_boost_data,
)
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test
# import wandb
import os
import numpy as np
import json


def bool_type_casting(x):
    return str(x).lower() in ("true", "1", "yes")


def define_argparser():
    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description="parser")

    ############### BASIC OPTION
    parser.add_argument(
        "--data_path", type=str, default="../data/", help="Data path를 설정할 수 있습니다."
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="./saved_models",
        help="Saved Model path를 설정할 수 있습니다.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "FM",
            "FFM",
            "DeepFFM",
            "NCF",
            "WDN",
            "DCN",
            "CNN_FM",
            "DeepCoNN",
            "CAT",
        ],
        help="학습 및 예측할 모델을 선택할 수 있습니다.",
    )
    parser.add_argument(
        "--data_shuffle",
        type=bool_type_casting,
        default=True,
        help="데이터 셔플 여부를 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Train/Valid split 비율을 조정할 수 있습니다."
    )
    parser.add_argument(
        "--stratify",
        type=bool_type_casting,
        default=False,
        help="rating에 따라 stratified split할지 여부를 조정합니다. default=0",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed 값을 조정할 수 있습니다.")
    parser.add_argument(
        "--use_best_model",
        type=bool_type_casting,
        default=True,
        help="검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.",
    )

    ############### TRAINING OPTION
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size를 조정할 수 있습니다."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epoch 수를 조정할 수 있습니다.")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning Rate를 조정할 수 있습니다."
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="RMSE",
        choices=["MSE", "RMSE"],
        help="손실 함수를 변경할 수 있습니다.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="ADAM",
        choices=["SGD", "ADAM"],
        help="최적화 함수를 변경할 수 있습니다.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.",
    )

    ############### GPU
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="학습에 사용할 Device를 조정할 수 있습니다.",
    )

    ############### FM, FFM, DeepFFM, NCF, WDN, DCN Common OPTION
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=16,
        help="FM, FFM, DeepFFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="DeepFFM, NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--mlp_dims",
        nargs="+",
        type=int,
        default=(16, 16),
        help="DeepFFM, NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.",
    )

    ############### FM, FFM, DeepFFM
    parser.add_argument(
        "--preprocessed",
        type=bool_type_casting,
        default=False,
        help="FM, FFM, DeepFFM에서 전처리된 데이터를 사용하는지 여부를 조정합니다. default=False",
    )

    ############### DeepFFM
    parser.add_argument(
        "--use_bn",
        type=bool_type_casting,
        default=True,
        help="DeepFFM에서 Batch Normalization 사용하는지 여부를 조정합니다. default=True",
    )

    ############### DCN
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="DCN에서 Cross Network의 레이어 수를 조정할 수 있습니다.",
    )

    ############### CNN_FM
    parser.add_argument(
        "--cnn_embed_dim",
        type=int,
        default=64,
        help="CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--cnn_latent_dim",
        type=int,
        default=12,
        help="CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.",
    )

    ############### DeepCoNN
    parser.add_argument(
        "--vector_create",
        type=bool_type_casting,
        default=False,
        help="DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.",
    )
    parser.add_argument(
        "--deepconn_embed_dim",
        type=int,
        default=32,
        help="DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--deepconn_latent_dim",
        type=int,
        default=10,
        help="DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--conv_1d_out_dim",
        type=int,
        default=50,
        help="DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--word_dim",
        type=int,
        default=768,
        help="DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.",
    )
    parser.add_argument(
        "--out_dim",
        type=int,
        default=32,
        help="DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.",
    )

    args = parser.parse_args()
    return args


def main(args):
    Setting.seed_everything(args.seed)

    ######################## DATA LOAD
    print(f"--------------- {args.model} Load Data ---------------")
    if args.model in ("FM", "FFM", "DeepFFM"):
        if args.preprocessed == True:
            print("preprocessed == True")
            data = context_data_load(args)
        if args.preprocessed == False:
            print("preprocessed == False")
            data = base_context_data_load(args)
    elif args.model in ("NCF", "WDN", "DCN"):
        data = dl_data_load(args)
    elif args.model == "CNN_FM":
        data = image_data_load(args)
    elif args.model == "DeepCoNN":
        import nltk

        nltk.download("punkt")
        data = text_data_load(args)
    else:
        pass

    ######################## Train/Valid Split
    print(f"--------------- {args.model} Train/Valid Split ---------------")
    if args.model in ("FM", "FFM", "DeepFFM"):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)

    elif args.model in ("NCF", "WDN", "DCN"):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.model == "CNN_FM":
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.model == "DeepCoNN":
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
    else:
        pass

    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)
    logger = Logger(args, log_path)
    logger.save_args()

    ######################## WanDB traker
    # wandb.init(
    #     project="project_name",
    #     name=f"{args.model} + {setting.save_time}",
    #     config=args,
    # )

    ######################## Model
    print(f"--------------- INIT {args.model} ---------------")
    model = models_load(args, data)

    # wandb.watch(model)

    ######################## TRAIN
    print(f"--------------- {args.model} TRAINING ---------------")
    model = train(args, model, data, logger, setting)

    ######################## INFERENCE
    print(f"--------------- {args.model} PREDICT ---------------")
    predicts = test(args, model, data, setting)

    ######################## SAVE PREDICT
    print(f"--------------- SAVE {args.model} PREDICT ---------------")
    if args.preprocessed == True:
        submission = pd.read_csv(args.data_path + "preprocessed/sample_submission.csv")
    elif args.preprocessed == False:
        submission = pd.read_csv(args.data_path + "sample_submission.csv")
    if args.model in (
        "FM",
        "FFM",
        "DeepFFM",
        "NCF",
        "WDN",
        "DCN",
        "CNN_FM",
        "DeepCoNN",
    ):
        submission["rating"] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


def init_cat_boost(args):
    assert args.model == "CAT"

    Setting.seed_everything(args.seed)
    setting = Setting()

    print("========CAT load data =========")
    data = load_cat_boost_data(args)

    # split
    print("========CAT split data =========")
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        data["train_df"].drop(["user_id", "isbn", "rating"], axis=1),
        data["train_df"]["rating"],
        test_size=0.2,
        random_state=42,
    )

    # param load and predict
    print("========CAT param load and injection =========")
    from catboost import CatBoostRegressor

    with open("./catboost_param.json") as f:
        cat_params = json.load(f)

    print("print params", cat_params)

    cbr = CatBoostRegressor(
        **cat_params,
        cat_features=list(
            X_train.drop(["age_bin", "year_of_publication"], axis=1).columns
        ),
        verbose=False,
    )

    print("========CAT fit and predict =========")
    cbr_study = cbr.fit(X_train, y_train)

    y_pred = cbr_study.predict(
        data["test_df"].drop(["user_id", "isbn", "rating"], axis=1)
    )

    from sklearn.metrics import mean_squared_error

    RMSE = mean_squared_error(y_val, cbr.predict(X_val), squared=False)

    # print("========wandb log =========")
    # wandb.init(
    #     project="level1_bookprediction_team",
    #     name=f"CAT + {setting.save_time}",
    #     group="CAT",
    #     config=cat_params,
    # )
    # wandb.log({"cat RMSE": RMSE})

    print("========CAT submission created... =========")
    submit = pd.DataFrame(
        {
            "user_id": np.array(data["test_df"].index),
            "isbn": data["test_df"]["isbn"],
            "rating": y_pred,
        }
    )
    from datetime import datetime

    now = datetime.now()

    os.makedirs("./submit", exist_ok=True)
    submit.to_csv("./submit/submission_" + str(now) + "_catboost.csv", index=False)

    print("========CAT done =========")


if __name__ == "__main__":
    args = define_argparser()
    # wandb_key=""
    # wandb.login(key=wandb_key)
    if args.model == "CAT":
        init_cat_boost(args)
    else:
        main(args)
