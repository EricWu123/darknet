#include "darknet.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};
// static char **names_classifier;
static network * net_classifier;
map_int_t label2int;

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //int num_count=0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand()%10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches)
	        {
		        dim = 618;
	        } 
            //int dim = (rand() % 4 + 16) * 32;
            if (dim >512)
            {
                dim = 512;
            }
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(loss>50000) loss=50000;
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_detector_recall(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/coco_val_5k.list");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

//just detect
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.3;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        // if (nms) do_nms_obj(dets, nboxes, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            //show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
/*
//detect and classify
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    // list *options = read_data_cfg(datacfg);
    // char *name_list = option_find_str(options, "names", "data/names.list");
    // char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1]; //the parameter of  the last layer


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        int i = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);

        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        char name[32] = "neg"; 
        char **name_ = (char **)malloc(sizeof(char*)*nboxes);;
        for(i = 0;i < nboxes;i++)
        {
            char* temp = (char*)malloc(32);
            name_[i] = temp;   
        }
        for(i = 0;i < nboxes;i++)
        {
            if(dets[i].prob[0] > thresh)
            {

                int x = (dets[i].bbox.x - dets[i].bbox.w/2) * im.w;
                int y = (dets[i].bbox.y - dets[i].bbox.h/2) * im.h;
                int w = dets[i].bbox.w * im.w;
                int h = dets[i].bbox.h * im.h;
                printf("%d %d %d %d %d %d\n", x,y,w,h,sized.w,sized.h);
                image crop_im = crop_image(im,x,y,w,h);
                
                predict_classifier_("cfg/tt100k_classifier.data","cfg/darknet19_tt100k.cfg","darknet19_tt100k_272.weights",name,crop_im);
                // printf("qqqq%s\n",name);
                strcpy(name_[i], name);       
                // name_[i] = name;
                // printf("tttttt%s\n",name_[i]);
                char temp_name[32] = "11111";
                sprintf(temp_name,"crop_image_%d",i);
                save_image(crop_im,temp_name);
                free_image(crop_im);
            }
        }
        // for(i = 0;i < nboxes;i++)
        // {
        //     printf("dddddd%s\n",name_[i]);
        // }
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        // if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        // draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        draw_detections_(im,dets,nboxes,thresh,name_,alphabet);  
        free_detections(dets, nboxes);
        for(i = 0;i < nboxes;i++)
        {
            char* temp = name_[i];
            free(temp);            
        }
        free(name_);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
*/

//just detect
void test_folder(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    DIR * dir;
    struct dirent *direntp; 

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    float nms=.35;

    while(1){
        char * input = (char *)malloc(sizeof(char) * 256);
        char * txt_ = (char *)malloc(sizeof(char) * 256);
        if(filename){
            strcpy(input, filename);
        } 
        dir = opendir(input);

        while ((direntp = readdir(dir)) != NULL)
        {  
            printf("file:%s\n", direntp->d_name);
            if(strcmp(direntp->d_name,".") == 0 || strcmp(direntp->d_name,"..") == 0) continue;
            char image_name[32];
            char s[32] = "1";
            strcpy(image_name,direntp->d_name);
            char * p = strchr(image_name,'.');
            int n = (int)(p - image_name);
            // char * s = (char *)malloc(sizeof(char) * 32);
            
            // printf("%ld\n", strlen(image_name));
            // printf("%d\n", n);
            strncpy(s,image_name,n);
            // free(image_name);
            
            char txt[256];
            // s = strncat(s,".txt",5);
            // printf("%s\n", s);
            sprintf(txt,"/home/cidi/PycharmProjects/test/Object-Detection-Metrics/detections/%s.txt",s);//create label txt file
            strcpy(txt_,txt);
            printf("%s\n", txt);
            FILE *fpWrite=fopen(txt_,"w");  
            if(fpWrite==NULL)  
            {  
                printf("%s\n", "open txt error");
                return;  
            }
            // printf("file:%s\n", direntp->d_name);
            // printf("%s\n",input);
            char buff[256];
            sprintf(buff,"%s%s",input,direntp->d_name);
            // strcpy(input_,buff);
            
            image im = load_image_color(buff,0,0);
            image sized = letterbox_image(im, net->w, net->h);

            layer l = net->layers[net->n-1]; //the parameter of  the last layer


            float *X = sized.data;
            time=what_time_is_it_now();
            network_predict(net, X);
            printf("%s: Predicted in %f seconds.\n", buff, what_time_is_it_now()-time);
            int nboxes = 0;
            int i = 0;
            detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
      
            if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);
            if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
            for(i = 0;i < nboxes;i++)
            {
                for(int j = 0; j < l.classes; ++j)
                {
                    thresh = 0.2;
                    if(dets[i].prob[j] > thresh)
                    {
                        int x = (dets[i].bbox.x - dets[i].bbox.w/2) * im.w;
                        int y = (dets[i].bbox.y - dets[i].bbox.h/2) * im.h;
                        int w = dets[i].bbox.w * im.w + x;
                        int h = dets[i].bbox.h * im.h + y;

                        if(x < 0) x = 0;
                        if(w > im.w-1) w = im.w-1;
                        if(y < 0) y = 0;
                        if(h > im.h-1) h = im.h-1;
                        fprintf(fpWrite, "%s %.2f %d %d %d %d\n", names[j],dets[i].prob[j],x,y,w,h);
                    }
                }
            }
            fclose(fpWrite);
            free_image(im);
            free_image(sized);
        }
    free(txt_);
    free(input);
    closedir(dir);
    break;
    }       
}

/*
// detect and classify
void test_folder(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    // list *options = read_data_cfg(datacfg);
    // char *name_list = option_find_str(options, "names", "data/names.list");
    // char **names = get_labels(name_list);
    DIR * dir;
    struct dirent *direntp; 

    // image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    // char buff[256];
    // char txt[256];
    // char * txt_ = txt;
    // char *input = buff;
    // char *input_ = buff;
    float nms=.45;

    // the classifier
    net_classifier = load_network("cfg/darknet19_tt100k.cfg","darknet19_tt100k_272.weights",0);
    set_batch_network(net_classifier, 1);
    list * options = read_data_cfg("cfg/tt100k_classifier.data");
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    names_classifier = get_labels(name_list);
    //the classifier end


    while(1){
        char * input = (char *)malloc(sizeof(char) * 256);
        char * txt_ = (char *)malloc(sizeof(char) * 256);
        if(filename){
            strcpy(input, filename);
        } 
        dir = opendir(input);

        while ((direntp = readdir(dir)) != NULL)
        {  
        printf("file:%s\n", direntp->d_name);
        if(strcmp(direntp->d_name,".") == 0 || strcmp(direntp->d_name,"..") == 0) continue;
        // char * image_name = direntp->d_name;
        // char * image_name = (char*)malloc(sizeof(char) * 32);
        char image_name[32];
        char s[32] = "1";
        strcpy(image_name,direntp->d_name);
        char * p = strchr(image_name,'.');
        int n = (int)(p - image_name);
        // char * s = (char *)malloc(sizeof(char) * 32);
        
        // printf("%ld\n", strlen(image_name));
        printf("%d\n", n);
        strncpy(s,image_name,n);
        // free(image_name);
        
        char txt[256];
        // s = strncat(s,".txt",5);
        printf("%s\n", s);
        sprintf(txt,"/home/cidi/TT100K/test_label/%s.txt",s);//create label txt file
        strcpy(txt_,txt);
        printf("%s\n", txt);
        FILE *fpWrite=fopen(txt_,"w");  
        if(fpWrite==NULL)  
        {  
            printf("%s\n", "open txt error");
            return;  
        }
        // printf("file:%s\n", direntp->d_name);
        // printf("%s\n",input);
        char buff[256];
        sprintf(buff,"%s%s",input,direntp->d_name);
        // strcpy(input_,buff);
        
        image im = load_image_color(buff,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1]; //the parameter of  the last layer


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", buff, what_time_is_it_now()-time);
        int nboxes = 0;
        int i = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);

        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        // char name[32] = "neg"; 
        // char **name_ = (char **)malloc(sizeof(char*)*nboxes);;
        // for(i = 0;i < nboxes;i++)
        // {
        //     char* temp = (char*)malloc(32);
        //     name_[i] = temp;   
        // }
        for(i = 0;i < nboxes;i++)
        {
            if(dets[i].prob[0] > thresh)
            {

                int x = (dets[i].bbox.x - dets[i].bbox.w/2) * im.w;
                int y = (dets[i].bbox.y - dets[i].bbox.h/2) * im.h;
                int w = dets[i].bbox.w * im.w;
                int h = dets[i].bbox.h * im.h;
                printf("%d %d %d %d %d %d\n", x,y,w,h,sized.w,sized.h);
                fprintf(fpWrite, "%s %d %d %d %d\n", s,x,y,w,h);
                // image crop_im = crop_image(im,x,y,w,h);
                
                // predict_classifier_demo(net_classifier,names_classifier,name,crop_im);
                // strcpy(name_[i], name);       
                // char temp_name[32] = "11111";
                // sprintf(temp_name,"crop_image_%d",i);
                // save_image(crop_im,temp_name);
                // free_image(crop_im);
            }
        }
        fclose(fpWrite);
        // free(s);
        
        // for(i = 0;i < nboxes;i++)
        // {
        //     printf("dddddd%s\n",name_[i]);
        // }
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        // if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        // draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        // draw_detections_(im,dets,nboxes,thresh,name_,alphabet);  
        // free_detections(dets, nboxes);
        // for(i = 0;i < nboxes;i++)
        // {
        //     char* temp = name_[i];
        //     free(temp);            
        // }
        // free(name_);
        // if(outfile){
        //     save_image(im, outfile);
        // }
        // else{
        //     save_image(im, "predictions");
// #ifdef OPENCV
//             cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
//             if(fullscreen){
//                 cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
//             }
//             // show_image(im, "predictions");
//             // cvWaitKey(10);
//             cvDestroyAllWindows();
// #endif
//         }

        free_image(im);
        free_image(sized);
        // if (filename) break;
    }
    free(txt_);
    free(input);
    closedir(dir);
    break;
    }
        

}
*/
void extract_feature(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{

    DIR * dir;
    struct dirent *direntp; 

    // the classifier
    net_classifier = load_network(cfgfile,weightfile,0);
    set_batch_network(net_classifier, 1);
    // list * options = read_data_cfg(datacfg);
    // char *name_list = option_find_str(options, "names", 0);
    // if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    // names_classifier = get_labels(name_list);
    //the classifier end


    while(1){
        char * input = (char *)malloc(sizeof(char) * 256);
        char * txt_ = (char *)malloc(sizeof(char) * 256);
        if(filename){
            strcpy(input, filename);
        } 
        dir = opendir(input);

        while ((direntp = readdir(dir)) != NULL)
        {  
        printf("file:%s\n", direntp->d_name);
        if(strcmp(direntp->d_name,".") == 0 || strcmp(direntp->d_name,"..") == 0) continue;
        // char * image_name = direntp->d_name;
        // char * image_name = (char*)malloc(sizeof(char) * 32);
        char image_name[32];
        char s[32] = "1";
        strcpy(image_name,direntp->d_name);
        // char * p = strchr(image_name,'.');
        char * p = strstr(image_name,".jpg");
        int n = (int)(p - image_name);
        strncpy(s,image_name,n); 
        char txt[256];
        sprintf(txt,"/home/cidi/darknet/data/feature_label/%s.txt",s);//create label txt file
        strcpy(txt_,txt);
        printf("%s\n", txt_);
        FILE *fpWrite=fopen(txt_,"w");  
        if(fpWrite==NULL)  
        {  
            printf("%s\n", "open txt error");
            return;  
        }
        
        char buff[256];
        sprintf(buff,"%s%s",input,direntp->d_name);
        printf("%s\n",buff);
        image im = load_image_color(buff,0,0);
        // image sized = letterbox_image(im, net->w, net->h);

        // time=what_time_is_it_now();

        // char name[32] = "neg"; 
        float features[1024] = {0};
        predict_classifier_demo_(net_classifier,im,features);
        // fprintf(fpWrite, "%s ", s);
        for(int i = 0;i < 1024;i++)
        {
            fprintf(fpWrite, "%f\n", features[i]);
        }

        fprintf(fpWrite, "\n");
        fclose(fpWrite);
        free_image(im);
    }
    free(txt_);
    free(input);
    closedir(dir);
    break;
    }
}
/*
light:0-49
lane:50-99
i:100-199
p:200-299
w:300-399
*/
void init_map()
{
    map_set(&label2int, "black", 0);
    map_set(&label2int, "red", 1);
    map_set(&label2int, "green", 3);
    map_set(&label2int, "yellow", 2);

    map_set(&label2int,"zhix",50);
    map_set(&label2int,"zuoz",51);
    map_set(&label2int, "youz", 52);
    map_set(&label2int, "diaot", 53);
    map_set(&label2int, "zxyz", 54);
    map_set(&label2int,"zxzz",55);
    map_set(&label2int,"zuoyou_z",56);
    map_set(&label2int, "lingx", 57);

    map_set(&label2int, "i2", 100);
    map_set(&label2int, "i3", 101);
    map_set(&label2int,"i4",102);
    map_set(&label2int,"i5",103);
    map_set(&label2int, "i10", 104);
    map_set(&label2int, "i16", 105);
    map_set(&label2int, "il60", 106);
    map_set(&label2int,"il80",107);
    map_set(&label2int,"il90",108);
    map_set(&label2int, "il100", 109);
    map_set(&label2int, "ipq", 110);
    map_set(&label2int, "is", 111);

    map_set(&label2int, "w13", 200);
    map_set(&label2int, "w22", 201);
    map_set(&label2int,"w32",202);
    map_set(&label2int,"w43",203);
    map_set(&label2int, "w55", 204);
    map_set(&label2int, "w57", 205);
    map_set(&label2int, "w58", 206);
    map_set(&label2int,"w59",207);

    map_set(&label2int,"p5",300);
    map_set(&label2int, "p6", 301);
    map_set(&label2int, "p9", 302);
    map_set(&label2int, "p10", 303);
    map_set(&label2int,"p11",304);
    map_set(&label2int,"p12",305);
    map_set(&label2int,"p16",306);
    map_set(&label2int, "p19", 307);
    map_set(&label2int, "p23", 308);
    map_set(&label2int, "p25", 309);
    map_set(&label2int,"p26",310);
    map_set(&label2int,"p27",311);
    map_set(&label2int,"p30",312);
    map_set(&label2int, "p31", 313);
    map_set(&label2int, "p_1", 314);
    map_set(&label2int, "p_3", 315);
    map_set(&label2int,"pa14",316);
    map_set(&label2int,"p_g",317);
    map_set(&label2int,"ph2.2",318);
    map_set(&label2int, "ph2.3", 319);
    map_set(&label2int, "ph4.5", 320);
    map_set(&label2int, "ph5", 321);
    map_set(&label2int,"pl20",322);
    map_set(&label2int,"pl30",323);
    map_set(&label2int,"pl40",324);
    map_set(&label2int, "pl50", 325);
    map_set(&label2int, "pl60", 326);
    map_set(&label2int, "pl70", 327);
    map_set(&label2int,"pl80",328);
    map_set(&label2int,"pl90",329);
    map_set(&label2int,"pl100",330);
    map_set(&label2int, "pl110", 331);
    map_set(&label2int, "pl120", 332);
    map_set(&label2int, "pl_5", 333);
    map_set(&label2int,"pm20",334);
    map_set(&label2int,"pm30",335);
    map_set(&label2int, "pm55", 336);
    map_set(&label2int, "p_n", 337);
    map_set(&label2int, "pne", 338);
    map_set(&label2int,"pr40",339);
    map_set(&label2int,"ps",340);

    map_set(&label2int,"neg",32);
}
void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    else if(0==strcmp(argv[2], "demo_metric")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo_metric(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    else if(0==strcmp(argv[2], "demo_3")) { // this is for detecting light, sign and lane.
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        char *datacfg_c = argv[6];//the classifier data　cfg
        char * cfg_c = argv[7];//the classifier cfg
        char * weights_c = argv[8];//the classifier weights
        char *filename = (argc > 9) ? argv[9]: 0;
        map_init(&label2int);
        init_map();
        demo_3(cfg, weights, datacfg_c,cfg_c,weights_c,thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    else if(0==strcmp(argv[2], "demo_3_v2")) { // this is for detecting light, sign and lane.the version 2
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        char datacfg_c[3][256];
        char cfg_c[3][256];
        char weights_c[3][256];
        for(int i = 0;i < 3;++i)
        {
            strcpy(datacfg_c[i],argv[6 + 3 * i]);
            strcpy(cfg_c[i],argv[7 + 3 * i]);
            strcpy(weights_c[i],argv[8 + 3 * i]);
        }
        char *filename = (argc > 15) ? argv[15]: 0;
        map_init(&label2int);
        init_map();
        demo_3_v2(cfg, weights, datacfg_c,cfg_c,weights_c,thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    else if(0 == strcmp(argv[2],"test_folder")) test_folder(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);// this function is like test_detector, but for image folder
    else if(0==strcmp(argv[2],"feature")) extract_feature(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen); // this function is like "test_detector", to get 1024-dim features and save to the files.
}
