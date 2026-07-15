#include "petpet_face/face_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace petpet_face
{

FaceTracker::FaceTracker(FaceTrackerConfig config)
    : config_(config)
{
    if (config_.minimumIou < 0.0F || config_.minimumIou > 1.0F ||
        config_.smoothingFactor <= 0.0F || config_.smoothingFactor > 1.0F ||
        config_.maximumMissedFrames < 0)
    {
        throw std::invalid_argument("Invalid face tracker configuration");
    }
}

std::vector<cv::Rect> FaceTracker::update(
    const std::vector<cv::Rect> &detections,
    const cv::Size &frameSize)
{
    std::vector<bool> detectionUsed(detections.size(), false);
    std::vector<bool> trackMatched(tracks_.size(), false);

    struct Match
    {
        std::size_t trackIndex;
        std::size_t detectionIndex;
        float iou;
    };
    std::vector<Match> matches;
    for (std::size_t trackIndex = 0; trackIndex < tracks_.size(); ++trackIndex)
    {
        for (std::size_t detectionIndex = 0;
             detectionIndex < detections.size(); ++detectionIndex)
        {
            const float iou = intersectionOverUnion(
                tracks_[trackIndex].box,
                cv::Rect2f(detections[detectionIndex]));
            if (iou >= config_.minimumIou)
            {
                matches.push_back({trackIndex, detectionIndex, iou});
            }
        }
    }
    std::sort(matches.begin(), matches.end(),
        [](const Match &first, const Match &second) {
            return first.iou > second.iou;
        });

    for (const Match &match : matches)
    {
        if (trackMatched[match.trackIndex] || detectionUsed[match.detectionIndex])
        {
            continue;
        }
        Track &track = tracks_[match.trackIndex];
        const cv::Rect2f measurement(detections[match.detectionIndex]);
        const float alpha = config_.smoothingFactor;
        const cv::Vec4f measuredVelocity(
            measurement.x - track.box.x,
            measurement.y - track.box.y,
            measurement.width - track.box.width,
            measurement.height - track.box.height);
        track.velocity = track.velocity * 0.5F + measuredVelocity * 0.5F;
        track.box.x = track.box.x * (1.0F - alpha) + measurement.x * alpha;
        track.box.y = track.box.y * (1.0F - alpha) + measurement.y * alpha;
        track.box.width = track.box.width * (1.0F - alpha) + measurement.width * alpha;
        track.box.height = track.box.height * (1.0F - alpha) + measurement.height * alpha;
        track.missedFrames = 0;
        trackMatched[match.trackIndex] = true;
        detectionUsed[match.detectionIndex] = true;
    }

    for (std::size_t index = 0; index < tracks_.size(); ++index)
    {
        if (!trackMatched[index])
        {
            Track &track = tracks_[index];
            if (!detections.empty())
            {
                track.missedFrames = config_.maximumMissedFrames + 1;
                continue;
            }
            track.box.x += track.velocity[0];
            track.box.y += track.velocity[1];
            track.box.width = std::max(2.0F, track.box.width + track.velocity[2]);
            track.box.height = std::max(2.0F, track.box.height + track.velocity[3]);
            track.velocity *= 0.8F;
            ++track.missedFrames;
        }
    }
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const Track &track) {
                return track.missedFrames > config_.maximumMissedFrames;
            }),
        tracks_.end());

    for (std::size_t index = 0; index < detections.size(); ++index)
    {
        if (!detectionUsed[index])
        {
            tracks_.push_back({cv::Rect2f(detections[index]), cv::Vec4f{}, 0});
        }
    }

    const cv::Rect frameBounds(0, 0, frameSize.width, frameSize.height);
    std::vector<cv::Rect> result;
    result.reserve(tracks_.size());
    for (const Track &track : tracks_)
    {
        cv::Rect box(
            static_cast<int>(std::lround(track.box.x)),
            static_cast<int>(std::lround(track.box.y)),
            static_cast<int>(std::lround(track.box.width)),
            static_cast<int>(std::lround(track.box.height)));
        box &= frameBounds;
        if (box.width > 1 && box.height > 1)
        {
            result.push_back(box);
        }
    }
    return result;
}

float FaceTracker::intersectionOverUnion(
    const cv::Rect2f &first,
    const cv::Rect2f &second)
{
    const cv::Rect2f intersection = first & second;
    if (intersection.width <= 0.0F || intersection.height <= 0.0F)
    {
        return 0.0F;
    }
    const float intersectionArea = intersection.area();
    const float unionArea = first.area() + second.area() - intersectionArea;
    return unionArea > 0.0F ? intersectionArea / unionArea : 0.0F;
}

} // namespace petpet_face
