#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static char **names_classifier;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static network * net_classifier;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;
int crop_i = 0; // used to count the num of the crop_image:detect_in_thread
// pthread_mutex_t mut; // used for thread synchronization


detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}
void detect_in_thread_()//this function is no thread
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    image display = copy_image(buff[buff_index]);
    float *X = buff_letter[buff_index].data;
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    // remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes);
    // dets = avg_predictions(net, &nboxes);
    //save video failed!!TAT
    // char  video_name[32] = "./data/sign.avi";
    // CvSize video_size  = cvSize(display.w,display.h);

    // CvVideoWriter * out = cvCreateVideoWriter(video_name,CV_FOURCC('D','I','V','X'),20, video_size,1); 
    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    char name[32] = "11111"; 
    int i = 0;
    char **name_ = (char **)malloc(sizeof(char*)*nboxes);;
    for(i = 0;i < nboxes;i++)
    {
        char* temp = (char*)malloc(32);
        name_[i] = temp;   
    }
    for(i = 0;i < nboxes;i++)
    {
        if(dets[i].prob[0] > demo_thresh)
        {
            int x = (dets[i].bbox.x - dets[i].bbox.w/2) * display.w;
            int y = (dets[i].bbox.y - dets[i].bbox.h/2) * display.h;
            int w = dets[i].bbox.w * display.w;
            int h = dets[i].bbox.h * display.h;
            // printf("%d %d %d %d %d %d\n", x,y,w,h,display.w,display.h);
            image crop_im = crop_image(display,x,y,w,h);
            
            predict_classifier_demo(net_classifier,names_classifier,name,crop_im);
            // printf("qqqq%s\n",name);

            strcpy(name_[i], name);       
            // name_[i] = name;
            // printf("tttttt%s\n",name_[i]);
            char temp_name[32] = "11111";
            sprintf(temp_name,"data/crop/crop_image_%d",crop_i);
            crop_i++;
            save_image(crop_im,temp_name);
            // IplImage * video_ipl = cvCreateImage(cvSize(display.w,display.h), IPL_DEPTH_8U, display.c);

            // cvWriteFrame(out,video_ipl);
            free_image(crop_im);
        }
    }
    for(i = 0;i < nboxes;i++)
    {
        printf("classifier:%s\n",name_[i]);
    }

    // printf("\033[2J");
    // printf("\033[1;1H");
    printf("FPS:%.1f\n",fps);
    printf("Objects:\n");
    printf("count:%d\n" ,crop_i);
    // image display1 = buff[buff_index];
    draw_detections_(display,dets,nboxes,demo_thresh,name_,demo_alphabet);
    // draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    // save_image(display1,"111111");
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    show_image(display, "Demo");
    cvWaitKey(1);
    free_image(display);

    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;

}
void *detect_in_thread(void *ptr)
{
    // pthread_mutex_lock(&mut);
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    image display = copy_image(buff[buff_index]);
    
    printf("detect:%d\n", buff_index);
    float *X = buff_letter[buff_index].data;
    printf("%s\n", "11111111111111");
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);
    //save video failed!!TAT
    // char  video_name[32] = "./data/sign.avi";
    // CvSize video_size  = cvSize(display.w,display.h);

    // CvVideoWriter * out = cvCreateVideoWriter(video_name,CV_FOURCC('D','I','V','X'),20, video_size,1); 
    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    char name[32] = "11111"; 
    int i = 0;
    char **name_ = (char **)malloc(sizeof(char*)*nboxes);;
    for(i = 0;i < nboxes;i++)
    {
        char* temp = (char*)malloc(32);
        name_[i] = temp;   
    }
    for(i = 0;i < nboxes;i++)
    {
        if(dets[i].prob[0] > demo_thresh)
        {
            int x = (dets[i].bbox.x - dets[i].bbox.w/2) * display.w;
            int y = (dets[i].bbox.y - dets[i].bbox.h/2) * display.h;
            int w = dets[i].bbox.w * display.w;
            int h = dets[i].bbox.h * display.h;
            // printf("%d %d %d %d %d %d\n", x,y,w,h,display.w,display.h);
            image crop_im = crop_image(display,x,y,w,h);
            
            predict_classifier_demo(net_classifier,names_classifier,name,crop_im);
            // printf("qqqq%s\n",name);

            strcpy(name_[i], name);       
            // name_[i] = name;
            // printf("tttttt%s\n",name_[i]);
            char temp_name[32] = "11111";
            sprintf(temp_name,"data/crop/crop_image_%d",crop_i);
            crop_i++;
            save_image(crop_im,temp_name);
            // IplImage * video_ipl = cvCreateImage(cvSize(display.w,display.h), IPL_DEPTH_8U, display.c);

            // cvWriteFrame(out,video_ipl);
            free_image(crop_im);
        }
    }
    for(i = 0;i < nboxes;i++)
    {
        printf("classifier:%s\n",name_[i]);
    }

    // printf("\033[2J");
    // printf("\033[1;1H");
    printf("FPS:%.1f\n",fps);
    printf("Objects:\n");
    printf("count:%d\n" ,crop_i);
    // image display1 = buff[buff_index];
    draw_detections_(display,dets,nboxes,demo_thresh,name_,demo_alphabet);
    // draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    // save_image(display1,"111111");
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    show_image(display, "Demo");
    cvWaitKey(0);
    free_image(display);
    printf("detect end:%d\n\n", buff_index);

    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;

    
    // pthread_mutex_unlock(&mut);
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    
    // pthread_mutex_lock(&mut);
    printf("fetch:%d\n", buff_index);
    // int status = fill_image_from_stream(cap, buff[buff_index]);
    // letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    // if(status == 0) demo_done = 1;
    printf("fetch end:%d\n", buff_index);
    // pthread_mutex_unlock(&mut);
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%2], "Demo", ipl);
    printf("show:%d\n", (buff_index + 1)%2);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    // pthread_t detect_thread;
    // pthread_t fetch_thread;

    srand(2222222);
    // the classifier
    net_classifier = load_network("cfg/darknet19_tt100k.cfg","darknet19_tt100k_272.weights",0);
    set_batch_network(net_classifier, 1);
    list * options = read_data_cfg("cfg/tt100k_classifier.data");
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    names_classifier = get_labels(name_list);
    //the classifier end
    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    // buff[1] = copy_image(buff[0]);
    // buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    // buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    // buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = what_time_is_it_now();

    // pthread_mutex_init(&mut,NULL);

    while(!demo_done){
        // buff_index = (buff_index +1) %2;
        buff_index = 0;
        
        // if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        detect_in_thread_();

        // printf("%s\n", "what happened");
        // if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");

        // pthread_join(detect_thread,0);
        // pthread_join(fetch_thread, 0);
        ++count;
        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            // display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%2], name);
        }
        
       
        ++count;
    }
}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

