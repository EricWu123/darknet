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
static char **names_classifier_v2[3];//for demo_3_v2
static image **demo_alphabet;//新的图片标签
static image **demo_alphabet_c; // 原来的字母标签
static int demo_classes;

static network *net;
static network * net_classifier;
static network * net_classifier_v2[3];
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0.5;
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
int save_count = 0;
char names_m[CLASS][32]; // for save labels
float features[CLASS * SAMPLES][1024]; //for save features of samples
// int crop_ii = 0;

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num,int letter);

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
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes,1);
    return dets;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    image display = buff[(buff_index+2) % 3];
    // image sized = buff_letter[(buff_index+2)%3];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    // remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    // dets = avg_predictions(net, &nboxes);
    dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes,1);
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

    char name[32] = "neg"; 
    int i = 0;
    // float thresh = 0.5;
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

            // int x = (dets[i].bbox.x - dets[i].bbox.w/2) * sized.w;
            // int y = (dets[i].bbox.y - dets[i].bbox.h/2) * sized.h;
            // int w = dets[i].bbox.w * sized.w;
            // int h = dets[i].bbox.h * sized.h;

            // !!you must make sure the ratio of sized'height and size'weight is the same as the ratio of the display(original).
            int x = (dets[i].bbox.x - dets[i].bbox.w/2) * display.w;
            int y = (dets[i].bbox.y - dets[i].bbox.h/2) * display.h;
            int w = dets[i].bbox.w * display.w;
            int h = dets[i].bbox.h * display.h;
            printf("%d %d %d %d %d %d\n", x,y,w,h,display.w,display.h);
            image crop_im = crop_image(display,x,y,w,h);
            
            predict_classifier_demo(net_classifier,names_classifier,name,crop_im,(float *)0);
            // printf("qqqq%s\n",name);

            strcpy(name_[i], name);       
            // name_[i] = name;
            // printf("tttttt%s\n",name_[i]);
            char temp_name[32] = "11111";
            sprintf(temp_name,"data/crop/crop_image_%d",crop_i);
            crop_i++;
            // save_image(display,temp_name);
            // IplImage * video_ipl = cvCreateImage(cvSize(display.w,display.h), IPL_DEPTH_8U, display.c);

            // cvWriteFrame(out,video_ipl);
            free_image(crop_im);
        }
    }
    // for(i = 0;i < nboxes;i++)
    // {
    //     printf("dddddd%s\n",name_[i]);
    // }
    // cvWaitKey(0);
    // printf("\033[2J");
    // printf("\033[1;1H");
    printf("FPS:%.1f\n",fps);
    printf("Objects:\n");
    printf("count:%d\n\n" ,crop_i);
    draw_detections_(display,dets,nboxes,demo_thresh,name_,demo_alphabet);
    // char save_name[32] = "1";
    // sprintf(save_name,"data/save/%d",save_count);
    // save_image(display,save_name);
    // save_count++;
    // image display = buff[(buff_index+2) % 3];
    // draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    // cvWaitKey(0);
    // int c = cvWaitKey(10);
    // if(c == 32)
    //   {
    //     cvWaitKey(10);
    //   }
    // free_image(sized);
    return 0;
}

void *detect_in_thread_metric(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    image display = buff[(buff_index+2) % 3];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    detection *dets = 0;
    int nboxes = 0;
    dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes,1);

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    char name[32] = "neg"; 
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
            printf("%d %d %d %d %d %d\n", x,y,w,h,display.w,display.h);
            image crop_im = crop_image(display,x,y,w,h);
            
            predict_classifier_demo(net_classifier,names_classifier,name,crop_im,(float *)0);
            int index = compare_feature(net_classifier,crop_im,features);
            strcpy(name,names_m[index]);
            printf("label:%s\n", name);

            strcpy(name_[i], name);       

            // char temp_name[32] = "11111";
            // sprintf(temp_name,"data/crop/crop_image_%d",crop_i);
            // crop_i++;
            // save_image(display,temp_name);
            free_image(crop_im);
        }
    }
    
    printf("FPS:%.1f\n",fps);
    printf("Objects:\n");
    printf("count:%d\n\n" ,crop_i);
    draw_detections_(display,dets,nboxes,demo_thresh,name_,demo_alphabet);
    // char save_name[32] = "1";
    // sprintf(save_name,"data/save/%d",save_count);
    // save_image(display,save_name);
    // save_count++;
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    // cvWaitKey(0);
    return 0;
}

void *detect_in_thread_3(void *ptr) // detect light, sign, and lane
{
    running = 1;
    float nms = .35;

    layer l = net->layers[net->n-1];
    image display = buff[(buff_index+2) % 3];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    detection *dets = 0;
    int nboxes = 0;
    dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes,1); 

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    char name[32] = "neg"; 
    int i = 0;
    char **name_ = (char **)malloc(sizeof(char*)*nboxes);//the boxes categories name
    for(i = 0;i < nboxes;i++)
    {
        char* temp = (char*)malloc(32);
        name_[i] = temp;   
    }
    for(i = 0;i < nboxes;i++)
    {
        for(int j = 0; j < l.classes; ++j)
        { 
            if(dets[i].prob[j] > demo_thresh)
            {

                // int x = (dets[i].bbox.x - dets[i].bbox.w/2) * sized.w;
                // int y = (dets[i].bbox.y - dets[i].bbox.h/2) * sized.h;
                // int w = dets[i].bbox.w * sized.w;
                // int h = dets[i].bbox.h * sized.h;


                // !!you must make sure the ratio of sized'height and sized'weight is the same as the ratio of the display(original).
                int x = (dets[i].bbox.x - dets[i].bbox.w/2) * display.w;
                int y = (dets[i].bbox.y - dets[i].bbox.h/2) * display.h;
                int w = dets[i].bbox.w * display.w;
                int h = dets[i].bbox.h * display.h;
                if(x < 0) x = 0;
                if(w > display.w-1) w = display.w-1;
                if(y < 0) y = 0;
                if(h > display.h-1) h = display.h-1;
                //printf("%d %d %d %d %d %d\n", x,y,w,h,display.w,display.h);
                // image crop_im = crop_image(display,x,y,w,h);   
                // predict_classifier_demo(net_classifier,names_classifier,name,crop_im);
                // printf("qqqq%s\n",name);
                strcpy(name_[i], name);


                /*******save the crop imgae *******/      
                // char temp_name[32] = "11111";
                // sprintf(temp_name,"data/crop/crop_image_%d",crop_i);
                // crop_i++;
                // save_image(crop_im,temp_name);
                // free_image(crop_im);
            }
        }
    }
    // for(i = 0;i < nboxes;i++)
    // {
    //     printf("name_:%s\n",name_[i]);
    // }

    // printf("\033[2J");
    // printf("\033[1;1H");
    printf("FPS:%.1f\n",fps);
    //printf("Objects:\n");
    //printf("count:%d\n\n" ,crop_i);

    // draw_detections_3(display,dets,nboxes,demo_thresh,name_,demo_alphabet,l.classes);

    /**** save the images the have detect rectangles****/
    // char save_name[32] = "1";
    // sprintf(save_name,"data/save/%d",save_count);
    // save_image(display,save_name);
    // save_count++;

    // cvWaitKey(0);
    free_detections(dets, nboxes);
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *detect_in_thread_3_v2(void *ptr) // detect light, sign, and lane
{
    running = 1;
    float nms = .35;

    layer l = net->layers[net->n-1];
    image display = buff[(buff_index+2) % 3];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    detection *dets = 0;
    int nboxes = 0;
    dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes,1); 

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    
    int i = 0;
    // char **name_ = (char **)malloc(sizeof(char*)*nboxes);//the boxes categories name
    // for(i = 0;i < nboxes;i++)
    // {
    //     char* temp = (char*)malloc(32);
    //     name_[i] = temp;   
    // }
    char **name_ = (char **)calloc(nboxes,sizeof(char*));//the boxes categories name
    for(i = 0;i < nboxes;i++)
    {
        char* temp = (char*)calloc(32,sizeof(char));
        name_[i] = temp;   
    }
    int nboxes_tmp = 1;
    if(nboxes > 0 )
        nboxes_tmp = nboxes;
    float confidence_c[nboxes_tmp];
    for(i = 0;i < nboxes_tmp;++i)
    {
        confidence_c[i] = 0.;
    }   
    demo_thresh = 0.2;
    for(i = 0;i < nboxes;i++)
    {
        float prob_max = dets[i].prob[0];
        int index = 0;
        for(int c = 1; c < l.classes; ++c)
        {
          if(dets[i].prob[c] > prob_max)
          {
            prob_max = dets[i].prob[c];
            index = c;
          }
        }
        // printf("index:%d %f\n", index,dets[i].prob[index]);
        if(dets[i].prob[index] > demo_thresh)
        {
          char name[32] = "neg";
          float * confidence_tmp = (float *)calloc(1,sizeof(float));
          int j = index;
          // int x = (dets[i].bbox.x - dets[i].bbox.w/2) * sized.w;
          // int y = (dets[i].bbox.y - dets[i].bbox.h/2) * sized.h;
          // int w = dets[i].bbox.w * sized.w;
          // int h = dets[i].bbox.h * sized.h;


          // !!you must make sure the ratio of sized'height and sized'weight is the same as the ratio of the display(original).
          int x = (dets[i].bbox.x - dets[i].bbox.w/2) * display.w;
          int y = (dets[i].bbox.y - dets[i].bbox.h/2) * display.h;
          int w = dets[i].bbox.w * display.w;
          int h = dets[i].bbox.h * display.h;
          if(x < 0) x = 0;
          if(w > display.w-1) w = display.w-1;
          if(y < 0) y = 0;
          if(h > display.h-1) h = display.h-1;
          //printf("%d %d %d %d %d %d\n", x,y,w,h,display.w,display.h);
          image crop_im = crop_image(display,x,y,w,h);   
          predict_classifier_demo(net_classifier_v2[j],names_classifier_v2[j],name,crop_im,confidence_tmp);
          // printf("name:%s\n",name);
          strcpy(name_[i], name);
          if(confidence_tmp != NULL)
            confidence_c[i] = *confidence_tmp;
          free(confidence_tmp);


          /*******save the crop imgae *******/      
        //   char temp_name[32] = "11111";
        //   sprintf(temp_name,"data/crop/crop_image_%d",crop_i);
        //   crop_i++;
        //   save_image(crop_im,temp_name);
          free_image(crop_im);
        }       
    }
    // for(i = 0;i < nboxes_tmp;++i)
    // {
    //     printf("%f\n",confidence_c[i]);
    // }
    // for(i = 0;i < nboxes;i++)
    // {
    //     printf("name_:%s\n",name_[i]);
    // }

    // printf("\033[2J");
    // printf("\033[1;1H");
    printf("FPS:%.1f\n",fps);
    //printf("Objects:\n");
    //printf("count:%d\n\n" ,crop_i);

    draw_detections_3(display,dets,nboxes,demo_thresh,name_,demo_alphabet,demo_alphabet_c,l.classes,confidence_c);

    /**** save the images the have detect rectangles****/
    // char save_name[32] = "1";
    // sprintf(save_name,"data/save/23000/%d",save_count);
    // save_image(display,save_name);
    // save_count++;

    cvWaitKey(0);
    free_detections(dets, nboxes);
    for(i = 0;i < nboxes;i++)
    {
        free(name_[i]);
    }
    free(name_);
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}


void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
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
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);
    // the classifier
    net_classifier = load_network("cfg/darknet19_tt100k.cfg","darknet19_tt100k_442.weights",0);
    set_batch_network(net_classifier, 1);
    list * options = read_data_cfg("cfg/tt100k_classifier.data");
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/label_classifier.list");
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
            printf("%f\n",(what_time_is_it_now() - demo_time));
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
void demo_metric(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
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
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);
    // the classifier
    net_classifier = load_network("cfg/darknet19_tt100k.cfg","darknet19_tt100k_443.weights",0);
    set_batch_network(net_classifier, 1);
    list * options = read_data_cfg("cfg/tt100k_classifier.data");
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    names_classifier = get_labels(name_list);

    //the classifier end

    // get labels
    FILE *pFile=fopen("/home/cidi/darknet/data/labels.txt","r");
    int names_flag = 0;
    if(pFile==NULL)  
    {  
        printf("%s\n", "open txt error");
        return;  
    }
    while(fscanf(pFile,"%[^\n]",names_m[names_flag])!=EOF)
    { 
        fgetc(pFile);
        names_flag++;
    }
    printf("%s\n", "labels end");
    // get labels end

    //get features
    for(int i = 0;i < CLASS;i++)
    {
        for(int j = 0;j < SAMPLES;j++)
        {
            // printf("%s","1111111111");
            char path[256];
            const char * path1;
            sprintf(path,"/home/cidi/darknet/data/feature_label/%s_%d.txt",names_m[i],j+1);
            path1 = path;
            // printf("%s\n",path1);
            FILE * file = fopen(path1,"r");
            int temp = 0;
            if(file == NULL)
            {
                printf("%s","open file error");
                return;
            }
            while(fscanf(file,"%f",&features[SAMPLES * i + j][temp++]) !=EOF);
        }
    }
    printf("%s\n", "features end");
    //get features end

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
        if(pthread_create(&detect_thread, 0, detect_in_thread_metric, 0)) error("Thread creation failed");
        if(!prefix){
            printf("%f\n",(what_time_is_it_now() - demo_time));
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

void demo_3(char *cfgfile, char *weightfile, char * datacfg_c, char * cfg_c,char * weights_c,
            float thresh, int cam_index, const char *filename, char **names, int classes, int delay, 
            char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet_3();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);
    // the classifier
    net_classifier = load_network(cfg_c,weights_c,0);
    set_batch_network(net_classifier, 1);
    list * options = read_data_cfg(datacfg_c);
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
        if(pthread_create(&detect_thread, 0, detect_in_thread_3, 0)) error("Thread creation failed");
        if(!prefix){
            printf("time:%f\n",(what_time_is_it_now() - demo_time));
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

void demo_3_v2(char *cfgfile, char *weightfile, char datacfg_c[3][256], char cfg_c[3][256],char weights_c[3][256],
            float thresh, int cam_index, const char *filename, char **names, int classes, int delay, 
            char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet_c = load_alphabet();
    image **alphabet = load_alphabet_3();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_alphabet_c = alphabet_c;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);
    // the classifier
    for(int i = 0;i < 3;++i)
    {
        net_classifier_v2[i] = load_network(cfg_c[i],weights_c[i],0);
        // printf("%s %s %s\n",cfg_c[i],weights_c[i],datacfg_c[i]);
        set_batch_network(net_classifier_v2[i], 1);
        list * options = read_data_cfg(datacfg_c[i]);
        char *name_list = option_find_str(options, "names", 0);
        if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
        names_classifier_v2[i] = get_labels(name_list);
    }
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
        if(pthread_create(&detect_thread, 0, detect_in_thread_3_v2, 0)) error("Thread creation failed");
        if(!prefix){
            printf("time:%f\n",(what_time_is_it_now() - demo_time));
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
void demo_metric(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
void demo_3(char *cfgfile, char *weightfile, char * datacfg_c, char * cfg_c,char * weights_c,
            float thresh, int cam_index, const char *filename, char **names, int classes, int delay, 
            char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
void demo_3_v2(char *cfgfile, char *weightfile, char datacfg_c[3][256], char cfg_c[3][256],char weights_c[3][256],
            float thresh, int cam_index, const char *filename, char **names, int classes, int delay, 
            char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

