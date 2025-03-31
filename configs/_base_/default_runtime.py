checkpoint_config = dict(interval=1)

log_config = dict(
    interval=400,
)
save_image_config = dict(
    interval=400,
)
# optimizer = dict(type='Adam', lr=0.0005)

loss = dict(type='MSELoss')

runner = dict(max_epochs=200)

checkpoints=None

resume=None