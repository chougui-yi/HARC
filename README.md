<<<<<<< HEAD
# HARC
=======
# HARC

HARC (Hash based Random Compression) is an acronym for a method of sorting malicious code families based on hash random compression

![model](https://github.com/chougui-yi/HARC/blob/master/assert/fig1.png)

# Good Case
![good case](https://github.com/chougui-yi/HARC/blob/master/assert/fig2.png)

# Model

1. We propose a malicious code data preprocessing method based on a specific hash algorithm. The method simulates the uneven data quality in the real world by screening and discarding the eigenvalues that contribute less to the model training.
2. In view of the possible effect of neural breakdown on classification performance, we propose a classifier based on autocorrelation matrix. This classifier uses the autocorrelation matrix of eigenvectors and combines the positive definite property of the autocorrelation matrix to prevent the problem of eigenvectors collapsing to a single class mean in traditional classification methods.

# Train
### Modify configuration file

```python
    train_loader, validate_loader = setDataLoader( 
        train_path, 
        test_path, 
        batch_size = args.batch_size, 
        token = args.token_type
    )
    trainer = Train(
        1,
        name = args.log_name,
        method_type= args.method,
        is_show=False,
        is_drop = args.is_drop,
        token_type = args.token_type,
    )
```

#### Run 
```shell
python run.py
```

>>>>>>> 1bbf1d0 (first commit)
